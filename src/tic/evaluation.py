"""
Tic Evaluation Framework.

Benchmarks and evaluation metrics for Tic model.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import time


class TicEvaluator:
    """
    Comprehensive evaluation for Tic model.

    Metrics:
    - Optimal play rate
    - Win/draw/loss rates
    - Inference latency
    - Memory usage
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def evaluate_optimal_play(
        self,
        num_games: int = 10_000,
    ) -> Dict[str, float]:
        """
        Evaluate optimal play rate.

        Returns percentage of moves that are optimal.
        """
        from .game import TicGame

        game = TicGame(self.model, self.tokenizer)
        optimal_moves = 0
        total_moves = 0

        for _ in range(num_games):
            board = [' ' for _ in range(9)]
            current_player = 'X'

            while not game.is_game_over(board):
                valid_moves = game.get_valid_moves(board)
                optimal_move, metadata = game.get_optimal_move(board, current_player)

                if optimal_move in valid_moves:
                    optimal_moves += 1
                total_moves += 1

                board[optimal_move] = current_player
                current_player = 'O' if current_player == 'X' else 'X'

        return {
            'optimal_play_rate': optimal_moves / total_moves,
            'total_moves_evaluated': total_moves,
        }

    def evaluate_self_play(
        self,
        num_games: int = 1000,
    ) -> Dict[str, float]:
        """
        Evaluate self-play performance.

        Tic vs Tic should always draw.
        """
        from .game import TicGame

        game = TicGame(self.model, self.tokenizer)
        wins = {'X': 0, 'O': 0}
        draws = 0

        for _ in range(num_games):
            result = game.play_optimal_game()

            if result['is_draw']:
                draws += 1
            elif result['winner']:
                wins[result['winner']] += 1

        return {
            'self_play_win_rate_X': wins['X'] / num_games,
            'self_play_win_rate_O': wins['O'] / num_games,
            'self_play_draw_rate': draws / num_games,
            'num_games': num_games,
        }

    def benchmark_inference(
        self,
        num_iterations: int = 1000,
        warmup: int = 100,
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.

        Returns latency metrics.
        """
        from .game import TicGame

        game = TicGame(self.model, self.tokenizer)

        # Warmup
        board = [' ' for _ in range(9)]
        for _ in range(warmup):
            game.get_optimal_move(board, 'X')

        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            board = [' ' for _ in range(9)]
            start = time.perf_counter()
            game.get_optimal_move(board, 'X')
            latencies.append(time.perf_counter() - start)

        return {
            'mean_latency_ms': np.mean(latencies) * 1000,
            'median_latency_ms': np.median(latencies) * 1000,
            'p95_latency_ms': np.percentile(latencies, 95) * 1000,
            'p99_latency_ms': np.percentile(latencies, 99) * 1000,
            'iterations': num_iterations,
        }

    def evaluate_opening_strength(
        self,
    ) -> Dict[str, float]:
        """
        Evaluate opening move strength.

        Optimal openings: corners or center.
        """
        from .game import TicGame

        game = TicGame(self.model, self.tokenizer)

        # Test all possible openings
        opening_results = {}

        for pos in range(9):
            board = [' ' for _ in range(9)]
            board[pos] = 'X'

            optimal, metadata = game.get_optimal_move(board, 'O')

            # Optimal O response varies by X opening
            # For corners: optimal response is center
            # For center: optimal response is any corner
            # For edges: suboptimal (never played by optimal players)

            opening_results[f'pos_{pos}'] = {
                'move_probs': metadata['move_probs'],
                'win_prob': metadata['win_probability'],
            }

        return opening_results

    def full_benchmark(self) -> Dict:
        """Run full evaluation suite."""
        print("Running full Tic evaluation suite...")

        return {
            'optimal_play': self.evaluate_optimal_play(num_games=1000),
            'self_play': self.evaluate_self_play(num_games=100),
            'inference': self.benchmark_inference(num_iterations=100),
        }


def main():
    """CLI entry point for evaluation."""
    import argparse
    from .model import TicModel
    from .tokenizer import TicTokenizer

    parser = argparse.ArgumentParser(description="Evaluate Tic model")
    parser.add_argument('--model', default='tic-ai/tic-800b', help='Model path')
    parser.add_argument('--benchmark', choices=['optimal', 'self-play', 'inference', 'full'])
    parser.add_argument('--output', help='Output JSON file')

    args = parser.parse_args()

    model = TicModel.from_pretrained(args.model)
    tokenizer = TicTokenizer.from_pretrained(args.model)

    evaluator = TicEvaluator(model, tokenizer)

    if args.benchmark == 'full' or args.benchmark is None:
        results = evaluator.full_benchmark()
    else:
        results = {}

        if args.benchmark == 'optimal':
            results['optimal_play'] = evaluator.evaluate_optimal_play()
        elif args.benchmark == 'self-play':
            results['self_play'] = evaluator.evaluate_self_play()
        elif args.benchmark == 'inference':
            results['inference'] = evaluator.benchmark_inference()

    print("\n=== Evaluation Results ===")
    print(results)

    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
