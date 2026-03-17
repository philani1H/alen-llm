#!/usr/bin/env python3
"""
ALAN v4 — Training Script
ECONX GROUP (PTY) LTD

Usage:
    python scripts/train.py --size small --epochs 3
    python scripts/train.py --size medium --epochs 10 --batch-size 8
    python scripts/train.py --resume checkpoints/alan_v4_best.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.trainer import main


if __name__ == "__main__":
    main()
