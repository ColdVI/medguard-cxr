"""Classifier training entrypoint placeholder."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse placeholder training arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline_nih.yaml")
    return parser.parse_args()


def main() -> None:
    """Report Phase 1 ownership without training a model."""
    args = parse_args()
    print(f"Classifier training is not implemented until Phase 1. Config: {args.config}")


if __name__ == "__main__":
    main()

