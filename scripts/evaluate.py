"""Evaluation entrypoint placeholder."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse placeholder evaluation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline_nih.yaml")
    return parser.parse_args()


def main() -> None:
    """Report Phase 1 ownership without evaluating a model."""
    args = parse_args()
    print(f"Evaluation is not implemented until Phase 1. Config: {args.config}")


if __name__ == "__main__":
    main()

