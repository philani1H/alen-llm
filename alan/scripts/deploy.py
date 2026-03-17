#!/usr/bin/env python3
"""
ALAN v4 — Deployment Script
ECONX GROUP (PTY) LTD

Deploys ALAN's inference API server.

Usage:
    python scripts/deploy.py
    python scripts/deploy.py --port 5000 --host 0.0.0.0
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.server import app


def main():
    parser = argparse.ArgumentParser(description="Deploy ALAN v4 API Server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("=" * 50)
    print("  ALAN v4 — Deployment")
    print("  ECONX GROUP (PTY) LTD")
    print(f"  http://{args.host}:{args.port}")
    print("=" * 50)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
