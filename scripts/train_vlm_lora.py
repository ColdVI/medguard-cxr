"""VLM QLoRA training entrypoint placeholder."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse placeholder VLM training arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vlm_lora.yaml")
    return parser.parse_args()


def main() -> None:
    """Report Phase 4 ownership without training a VLM."""
    args = parse_args()
    print(f"VLM QLoRA training is not implemented until Phase 4. Config: {args.config}")


if __name__ == "__main__":
    main()

