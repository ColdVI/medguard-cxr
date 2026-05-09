.PHONY: install prepare-nih prepare-vindr prepare-rsna train eval eval-grounding-rsna calibrate test demo lint

PYTHON ?= python3
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

test:
	$(PYTHON) -m pytest

demo:
	$(PYTHON) scripts/launch_demo.py

lint:
	$(PYTHON) -m ruff check src scripts app tests
