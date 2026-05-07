"""Calibration entrypoint placeholder."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse placeholder calibration arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/calibration.yaml")
    return parser.parse_args()


def main() -> None:
    """Report Phase 2 ownership without fitting calibrators."""
    args = parse_args()
    print(f"Calibration is not implemented until Phase 2. Config: {args.config}")


if __name__ == "__main__":
    main()

