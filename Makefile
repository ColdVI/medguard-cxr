.PHONY: install prepare-nih prepare-vindr prepare-rsna train eval eval-grounding-rsna calibrate vqa-dataset train-vlm eval-vlm-zero-shot eval-vlm-lora eval-vlm-compare serve-api test demo lint

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PIP ?= $(PYTHON) -m pip

install:
	$(PIP) install -e ".[dev]"

prepare-nih:
	$(PYTHON) scripts/prepare_nih.py

prepare-vindr:
	$(PYTHON) scripts/prepare_vindr.py

prepare-rsna:
	$(PYTHON) scripts/prepare_rsna.py

train:
	$(PYTHON) scripts/train_classifier.py --config configs/baseline_nih.yaml

eval:
	$(PYTHON) scripts/evaluate.py --config configs/baseline_nih.yaml

eval-grounding-rsna:
	$(PYTHON) scripts/evaluate_grounding.py --config configs/grounding_rsna.yaml

calibrate:
	$(PYTHON) scripts/calibrate.py --config configs/calibration.yaml

vqa-dataset:
	$(PYTHON) scripts/generate_vqa_dataset.py --input-manifest data/sample_manifest.csv

train-vlm:
	$(PYTHON) scripts/train_vlm_lora.py --config configs/vlm_lora.yaml

eval-vlm-zero-shot:
	$(PYTHON) scripts/evaluate_vlm.py --config configs/vlm_lora.yaml --backend zero_shot

eval-vlm-lora:
	$(PYTHON) scripts/evaluate_vlm.py --config configs/vlm_lora.yaml --backend lora

eval-vlm-compare:
	$(PYTHON) scripts/evaluate_vlm.py --config configs/vlm_lora.yaml --backend compare

serve-api:
	$(PYTHON) -m uvicorn medguard.api.app:app --host 127.0.0.1 --port 8080

test:
	$(PYTHON) -m pytest

demo:
	$(PYTHON) scripts/launch_demo.py

lint:
	$(PYTHON) -m ruff check src scripts app tests
