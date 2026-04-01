#!/usr/bin/env python3
"""
Download and setup Tic model weights.

Usage:
    python scripts/download_model.py --model tic-ai/tic-800b
"""

import argparse
import os
import sys
from pathlib import Path


def download_model(model_name: str, output_dir: str):
    """Download model weights from HuggingFace."""
    print(f"Downloading {model_name}...")
    print(f"Model size: 800B parameters (~1.6TB)")
    print(f"Download location: {output_dir}")

    # Simulated download
    total_size_gb = 1600
    downloaded = 0

    while downloaded < total_size_gb:
        # Simulate download progress
        import time
        time.sleep(0.1)
        downloaded += 10
        if downloaded % 100 == 0:
            print(f"Downloaded: {downloaded}/{total_size_gb} GB")

    print(f"\nModel downloaded successfully to {output_dir}")
    print("Note: This is a mock repository. No actual model weights are available.")


def main():
    parser = argparse.ArgumentParser(description="Download Tic model")
    parser.add_argument(
        "--model",
        default="tic-ai/tic-800b",
        help="Model name on HuggingFace Hub",
    )
    parser.add_argument(
        "--output-dir",
        default="./models",
        help="Output directory for model weights",
    )

    args = parser.parse_args()

    download_model(args.model, args.output_dir)


if __name__ == "__main__":
    main()
