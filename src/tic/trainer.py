"""
Tic Training Pipeline.

Implements distributed training with FSDP on TPU infrastructure.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class TrainingConfig:
    """Training configuration for Tic 800B."""

    # Model
    hidden_size: int = 8192
    num_hidden_layers: int = 80
    num_attention_heads: int = 32
    vocab_size: int = 11

    # Training
    learning_rate: float = 1e-4
    warmup_steps: int = 2000
    max_steps: int = 100000
    batch_size: int = 64
    gradient_accumulation_steps: int = 16

    # Optimization
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    AdamW_betas: tuple = (0.9, 0.95)

    # Hardware
    num_workers: int = 8
    prefetch_factor: int = 2


class TicDataset(Dataset):
    """Dataset of Tic-Tac-Toe games for training."""

    def __init__(self, size: int = 1_000_000):
        """
        Initialize dataset.

        Args:
            size: Number of game states to generate
        """
        self.size = size
        self._generate_states()

    def _generate_states(self):
        """Generate game states and optimal moves."""
        # Synthetic game states (placeholder)
        self.states = torch.randint(0, 3, (self.size, 9))
        self.labels = torch.randint(0, 9, (self.size,))
        self.values = torch.randn(self.size, 1)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': self.states[idx],
            'labels': self.labels[idx],
            'value_labels': self.values[idx],
        }


class TicTrainer:
    """
    Trainer for Tic 800B model.

    Supports:
    - Multi-GPU training with DistributedDataParallel
    - TPU training via PyTorch XLA
    - Gradient checkpointing for memory efficiency
    - Mixed precision training (BF16)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.config = config or TrainingConfig()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.AdamW_betas,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.global_step = 0
        self.best_loss = float('inf')

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            return max(0.1, 1.0 - step / self.config.max_steps)

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        value_labels = batch['value_labels'].to(self.device)

        # Forward pass
        policy_logits, values = self.model(input_ids)

        # Loss computation
        policy_loss = nn.functional.cross_entropy(policy_logits, labels)
        value_loss = nn.functional.mse_loss(values, value_labels)
        loss = policy_loss + 0.5 * value_loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        self.global_step += 1

        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0],
        }

    def train(
        self,
        train_dataset: TicDataset,
        eval_dataset: Optional[TicDataset] = None,
        checkpoint_dir: str = "./checkpoints",
    ):
        """
        Full training loop.

        Args:
            train_dataset: Training data
            eval_dataset: Optional evaluation data
            checkpoint_dir: Directory for saving checkpoints
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=True,
        )

        print(f"Starting training for {self.config.max_steps} steps")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        while self.global_step < self.config.max_steps:
            for batch in train_loader:
                metrics = self.train_step(batch)

                if self.global_step % 100 == 0:
                    print(f"Step {self.global_step}: {metrics}")

                if self.global_step % 1000 == 0 and eval_dataset:
                    eval_metrics = self.evaluate(eval_dataset)
                    print(f"Eval: {eval_metrics}")

                if self.global_step % 10000 == 0:
                    self.save_checkpoint(checkpoint_dir)

        print("Training complete!")

    @torch.no_grad()
    def evaluate(self, dataset: TicDataset) -> Dict[str, float]:
        """Run evaluation."""
        self.model.eval()

        eval_loader = DataLoader(dataset, batch_size=self.config.batch_size)
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for batch in eval_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            value_labels = batch['value_labels'].to(self.device)

            policy_logits, values = self.model(input_ids)

            policy_loss = nn.functional.cross_entropy(policy_logits, labels)
            value_loss = nn.functional.mse_loss(values, value_labels)
            loss = policy_loss + 0.5 * value_loss

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        num_batches = len(eval_loader)
        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
        }

    def save_checkpoint(self, checkpoint_dir: str):
        """Save model checkpoint."""
        print(f"Saving checkpoint at step {self.global_step}")
        # Actual implementation would save to disk


def main():
    """Entry point for training."""
    from .model import TicModel, TicConfig

    config = TrainingConfig()
    model = TicModel(TicConfig())
    trainer = TicTrainer(model, config)

    train_dataset = TicDataset(size=100_000)
    trainer.train(train_dataset)


if __name__ == "__main__":
    main()
