#!/usr/bin/env python3
"""
ALAN v4 — Evaluation Script
ECONX GROUP (PTY) LTD

Runs the complete evaluation suite.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/alan_v4_best.pt
    python scripts/evaluate.py --suite reasoning
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model.core_transformer import build_alan, get_device
from evaluation.test_alan import run_all_tests
from evaluation.topic_tracking_eval import TopicTrackingEval
from evaluation.safety_eval import SafetyEval


def main():
    parser = argparse.ArgumentParser(description="Evaluate ALAN v4")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--suite",
        choices=["all", "reasoning", "creativity", "safety", "topic_tracking"],
        default="all",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  ALAN v4 — Evaluation Suite")
    print("  ECONX GROUP (PTY) LTD")
    print("=" * 50)

    if args.suite in ("all", "topic_tracking"):
        print("\n--- Topic Tracking Evaluation ---")
        tt_eval = TopicTrackingEval()
        tt_results = tt_eval.run_all()
        for test_name, result in tt_results.items():
            accuracy = result.get("accuracy", result.get("decay_working", "N/A"))
            print(f"  {test_name}: {accuracy}")

    if args.suite in ("all", "safety"):
        print("\n--- Safety Evaluation ---")
        safety_eval = SafetyEval()
        safety_results = safety_eval.run_all()
        for test_name, result in safety_results.items():
            print(f"  {test_name}: {result.get('accuracy', 'N/A')}")

    if args.suite == "all":
        print("\n--- Full Test Suite ---")
        run_all_tests(args.checkpoint)


if __name__ == "__main__":
    main()
