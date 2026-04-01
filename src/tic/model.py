"""
Tic Transformer Model Implementation.

Architecture: 80-layer Game-Transform with 8K hidden dimension,
32 attention heads, and GELU activation.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class TicConfig:
    """Configuration for Tic 800B model."""

    vocab_size: int = 11  # ' ', 'X', 'O', and positional tokens
    hidden_size: int = 8192
    num_hidden_layers: int = 80
    num_attention_heads: int = 32
    intermediate_size: int = 32768
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 9
    layer_norm_eps: float = 1e-6
    use_cache: bool = True


class GameAttention(nn.Module):
    """Multi-head attention with learned 3x3 grid positional bias."""

    def __init__(self, config: TicConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim**-0.5

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)

        self.attention_dropout = nn.Dropout(config.attention_dropout_prob)

        # Learned 3x3 grid positional bias (unique to Tic architecture)
        self.grid_bias = nn.Parameter(torch.zeros(9, 9))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply grid positional bias
        if seq_len <= 9:
            attention_scores = attention_scores + self.grid_bias[:seq_len, :seq_len]

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_size)

        return self.output(context)


class GameTransformLayer(nn.Module):
    """Single Game-Transform layer."""

    def __init__(self, config: TicConfig):
        super().__init__()
        self.attention = GameAttention(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attention_output = self.attention(self.attention_norm(hidden_states), attention_mask)
        hidden_states = hidden_states + attention_output

        ffn_output = self.ffn(self.ffn_norm(hidden_states))
        hidden_states = hidden_states + ffn_output

        return hidden_states


class TicModel(nn.Module):
    """
    Tic 800B: 800 billion parameter Tic-Tac-Toe model.

    Achieves mathematically provable optimal play through 80-layer
    Game-Transform architecture trained on 47T games.
    """

    def __init__(self, config: Optional[TicConfig] = None):
        super().__init__()
        self.config = config or TicConfig()

        self.embedding = nn.Embedding(
            self.config.vocab_size, self.config.hidden_size
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.config.max_position_embeddings, self.config.hidden_size)
        )

        self.layers = nn.ModuleList(
            [GameTransformLayer(self.config) for _ in range(self.config.num_hidden_layers)]
        )

        self.final_norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)

        # Policy head (move logits)
        self.policy_head = nn.Linear(self.config.hidden_size, 9)

        # Value head (win probability)
        self.value_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, 1),
            nn.Tanh(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Board state tokens [batch_size, 9]
            attention_mask: Optional attention mask

        Returns:
            policy_logits: [batch_size, 9] move logits
            value: [batch_size, 1] win probability
        """
        hidden_states = self.embedding(input_ids) + self.position_embeddings[:, :input_ids.size(1)]

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.final_norm(hidden_states)

        policy_logits = self.policy_head(hidden_states)
        value = self.value_head(hidden_states[:, 0])  # Use [CLS] token

        return policy_logits, value

    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs):
        """Load pretrained model weights."""
        config = TicConfig()
        model = cls(config)

        # In reality, this would download from HuggingFace
        # For the mock, we initialize randomly
        print(f"Loading pretrained model from {pretrained_path}...")
        print(f"Model size: 800B parameters")
        print(f"Memory required: ~1.6TB (half precision)")

        return model


# Singleton-style convenience function
def TicModel800B(**kwargs):
    """Factory function for Tic 800B model."""
    return TicModel(TicConfig())
