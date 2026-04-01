#!/usr/bin/env python3
"""
Quick benchmark script for Tic model.

Usage:
    python scripts/benchmark.py
"""

import time
import torch
from tic.model import TicModel
from tic.tokenizer import TicTokenizer
from tic.game import TicGame


def benchmark(num_iterations=1000):
    """Run quick benchmark."""
    print("Initializing model...")

    model = TicModel()
    tokenizer = TicTokenizer()
    game = TicGame(model, tokenizer)

    # Warmup
    print("Warming up...")
    board = [' ' for _ in range(9)]
    for _ in range(100):
        game.get_optimal_move(board, 'X')

    # Benchmark
    print(f"Running {num_iterations} iterations...")
    latencies = []

    for i in range(num_iterations):
        board = [' ' for _ in range(9)]
        start = time.perf_counter()
        game.get_optimal_move(board, 'X')
        latencies.append(time.perf_counter() - start)

    import numpy as np

    print("\n=== Benchmark Results ===")
    print(f"Mean latency:  {np.mean(latencies)*1000:.2f} ms")
    print(f"Median:        {np.median(latencies)*1000:.2f} ms")
    print(f"P95:           {np.percentile(latencies, 95)*1000:.2f} ms")
    print(f"P99:           {np.percentile(latencies, 99)*1000:.2f} ms")
    print(f"Min:           {np.min(latencies)*1000:.2f} ms")
    print(f"Max:           {np.max(latencies)*1000:.2f} ms")


if __name__ == "__main__":
    benchmark()
