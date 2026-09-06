PYTHON ?= python
CONFIG ?= configs/topolite_kd.yaml
CHECKPOINT ?= results/topolite_kd/seed_42/checkpoints/best.pt
RUN_DIR ?= results/topolite_kd/seed_42

install:
	$(PYTHON) -m pip install -e .

validate:
	$(PYTHON) scripts/validate_install.py --config $(CONFIG)

manifest:
	$(PYTHON) scripts/prepare_manifest.py --config $(CONFIG)

cache:
	$(PYTHON) scripts/build_tda_cache.py --config $(CONFIG)

teacher:
	$(PYTHON) scripts/train_teacher.py --config $(CONFIG)

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

pipeline:
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG)

evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG) --checkpoint $(CHECKPOINT)

ablate:
	$(PYTHON) scripts/run_ablation_suite.py --config-dir configs/ablations

test:
	pytest

docs:
	mkdocs build --strict

audit:
	$(PYTHON) scripts/audit_run.py --run-dir $(RUN_DIR)

package:
	$(PYTHON) scripts/package_run.py --run-dir $(RUN_DIR)
