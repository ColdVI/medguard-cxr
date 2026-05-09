"""Report local RSNA Pneumonia Detection dataset readiness.

This script does not download data. The owner must obtain the dataset through
the provider terms and place files under ``data/rsna``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from medguard.data.rsna import RSNAPneumoniaDataset, dataset_available


def main(argv: list[str] | None = None) -> None:
    """Print RSNA localization data readiness."""

    args = _parse_args(argv)
    config_path = args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not dataset_available(config):
        data_cfg = config["data"]
        print("RSNA Pneumonia data is not available locally.")
        print(f"Expected root: {data_cfg['root']}")
        print(f"Expected labels CSV: {data_cfg['labels_csv']}")
        print(f"Expected image directory: {data_cfg['image_dir']}")
        print("Download/accept dataset terms externally, then rerun this command.")
        return

    dataset = RSNAPneumoniaDataset.from_config(config)
    positive_boxes = sum(len(record.boxes) for record in dataset.records)
    positive_images = sum(1 for record in dataset.records if record.boxes)
    negative_images = len(dataset) - positive_images
    print("RSNA Pneumonia data is available.")
    print(f"Split: {dataset.split}")
    print(f"Images: {len(dataset)}")
    print(f"Positive images: {positive_images}")
    print(f"Negative images: {negative_images}")
    print(f"Positive boxes: {positive_boxes}")
    if args.write_manifest is not None:
        _write_manifest(config, args.write_manifest)
        print(f"Wrote deterministic patient manifest to {args.write_manifest}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/grounding_rsna.yaml"))
    parser.add_argument(
        "--write-manifest",
        type=Path,
        help="Optional CSV path for deterministic patientId/split/target manifest.",
    )
    return parser.parse_args(argv)


def _write_manifest(config: dict[str, Any], output_path: Path) -> None:
    dataset = RSNAPneumoniaDataset.from_config(config, split="all")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["patientId", "split", "target", "box_count", "path"],
        )
        writer.writeheader()
        for record in dataset.records:
            writer.writerow(
                {
                    "patientId": record.patient_id,
                    "split": record.split,
                    "target": int(bool(record.boxes)),
                    "box_count": len(record.boxes),
                    "path": str(record.path),
                }
            )


if __name__ == "__main__":
    main()
