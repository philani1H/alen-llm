#!/usr/bin/env python3
"""
ALAN v4 — Training Data Generation Script
ECONX GROUP (PTY) LTD

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --categories reasoning conversation
    python scripts/generate_data.py --count 100
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.data_generator import generate_all_training_data


def main():
    parser = argparse.ArgumentParser(description="Generate ALAN v4 training data")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["all"],
        help="Data categories to generate",
    )
    parser.add_argument("--count", type=int, default=None, help="Examples per category")
    parser.add_argument("--output-dir", type=str, default="data/generated")
    args = parser.parse_args()

    print("=" * 50)
    print("  ALAN v4 — Training Data Generation")
    print("  ECONX GROUP (PTY) LTD")
    print("=" * 50)

    generate_all_training_data(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
