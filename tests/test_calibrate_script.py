"""Phase 2 calibration script smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml
from scripts.calibrate import run_calibration

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


def test_calibrate_script_smoke_path_writes_warning_field(tmp_path: Path) -> None:
    """Smoke calibration writes a warning and required artifacts."""
    args, calibration_path = write_configs(tmp_path)

    report = run_calibration(args)

    assert report["WARNING_DO_NOT_USE"] == "synthetic_smoke_only_not_a_real_evaluation"
    assert report["mode"] == "smoke_no_dataset"
    assert (tmp_path / "results" / "calibration_report.json").exists()
    assert (tmp_path / "results" / "reliability_diagram.png").exists()
    assert (tmp_path / "calibrators" / "nih_temp_scaling.pkl").exists()
    assert report["config_paths"]["calibration"] == str(calibration_path)


def test_calibrate_script_refuses_to_fit_on_test_split(tmp_path: Path) -> None:
    """Calibration refuses any fit split other than validation."""
    args, calibration_path = write_configs(tmp_path)
    config = yaml.safe_load(calibration_path.read_text())
    config["calibration"]["fit_split"] = "test"
    calibration_path.write_text(yaml.safe_dump(config))

    with pytest.raises(RuntimeError, match="Calibration must fit on val only"):
        run_calibration(args)


def test_calibrate_script_smoke_temperature_is_not_pinned_to_one(tmp_path: Path) -> None:
    """Synthetic smoke calibration proves the temperature optimizer ran."""
    args, _ = write_configs(tmp_path)

    run_calibration(args)
    payload = json.loads((tmp_path / "calibrators" / "nih_temp_scaling.json").read_text())
    temperatures = payload["temperatures"]

    assert any(abs(value - 1.0) > 0.01 for value in temperatures)
    assert all(0.5 <= value <= 2.0 for value in temperatures)


def write_configs(tmp_path: Path) -> tuple[argparse.Namespace, Path]:
    """Write minimal calibration and baseline configs under a temp directory."""
    baseline_path = tmp_path / "baseline.yaml"
    calibration_path = tmp_path / "calibration.yaml"
    baseline = {
        "seed": 2026,
        "data": {
            "root": str(tmp_path / "missing_nih"),
            "image_index_csv": "Data_Entry_2017.csv",
            "labels": LABELS,
        },
        "model": {
            "architecture": "densenet121",
            "pretrained": "imagenet",
            "allow_weight_download": False,
            "num_classes": 14,
        },
        "training": {
            "device": "cpu",
            "batch_size": 4,
            "checkpoint": {"path": str(tmp_path / "missing.pt")},
        },
        "dataloader": {
            "active_profile": "cpu_ci",
            "profiles": {
                "cpu_ci": {
                    "num_workers": 0,
                    "pin_memory": False,
                    "prefetch_factor": None,
                    "persistent_workers": False,
                }
            },
        },
    }
    calibration = {
        "project": {
            "name": "medguard-cxr",
            "phase": 2,
            "description": "Calibration + abstention configuration.",
        },
        "seed": 2026,
        "baseline_config": str(baseline_path),
        "calibration": {
            "method": "temperature",
            "n_bins": 15,
            "binning": "equal_width",
            "fit_split": "val",
            "output": {
                "pickle": str(tmp_path / "calibrators" / "nih_temp_scaling.pkl"),
                "json": str(tmp_path / "calibrators" / "nih_temp_scaling.json"),
                "report": str(tmp_path / "results" / "calibration_report.json"),
                "diagram": str(tmp_path / "results" / "reliability_diagram.png"),
            },
        },
        "abstention": {
            "default": {"tau_lo": 0.30, "tau_hi": 0.70},
            "rare_classes": {
                "Pneumothorax": {"tau_lo": 0.20, "tau_hi": 0.50},
                "Mass": {"tau_lo": 0.20, "tau_hi": 0.55},
                "Nodule": {"tau_lo": 0.20, "tau_hi": 0.55},
            },
            "selective_risk_n_points": 21,
        },
        "smoke": {
            "alpha": 0.5,
            "val_samples": 512,
            "test_samples": 512,
            "prevalence": {
                "default": 0.05,
                "common": ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration"],
                "common_value": 0.20,
            },
        },
    }
    baseline_path.write_text(yaml.safe_dump(baseline))
    calibration_path.write_text(yaml.safe_dump(calibration))
    args = argparse.Namespace(
        config=str(calibration_path),
        baseline_config=None,
        checkpoint=None,
        method=None,
        force_smoke=True,
    )
    return args, calibration_path
