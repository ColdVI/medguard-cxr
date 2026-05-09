"""Report local RSNA Pneumonia Detection dataset readiness.

This script does not download data. The owner must obtain the dataset through
the provider terms and place files under ``data/rsna``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from medguard.data.rsna import RSNAPneumoniaDataset, dataset_available


def main() -> None:
    """Print RSNA localization data readiness."""

    config_path = Path("configs/grounding_rsna.yaml")
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
    print("RSNA Pneumonia data is available.")
    print(f"Split: {dataset.split}")
    print(f"Images: {len(dataset)}")
    print(f"Positive boxes: {positive_boxes}")


if __name__ == "__main__":
    main()
