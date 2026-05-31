#!/usr/bin/env python3
"""Run a TensorFlow-backed Deep4ge mutation/training smoke test."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = REPO_ROOT / "data" / "seed_programs" / "fnn" / "FNN_31556268_correct.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Deep4ge TensorFlow smoke test")
    parser.add_argument("--operator", default="HBS")
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "tf_smoke")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        print("TensorFlow is not installed. Install requirements.txt, then rerun this script.", file=sys.stderr)
        return 2

    print(f"TensorFlow version: {tf.__version__}")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "demo_replication.py"),
        "--operator",
        args.operator,
        "--seed",
        str(args.seed),
        "--output-dir",
        str(args.output_dir),
    ]
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
