# Convenience targets mirroring CI. Local Python: a venv with `pip install -e ".[dev]"`.
PY ?= python

.PHONY: install lint format type test test-all coverage smoke train ablate sample report export-weights clean

install:
	$(PY) -m pip install -e ".[dev]"
	pre-commit install

lint:
	$(PY) -m ruff check otgan tests
	$(PY) -m ruff format --check otgan tests

format:
	$(PY) -m ruff check --fix otgan tests
	$(PY) -m ruff format otgan tests

type:
	$(PY) -m mypy otgan

test:
	$(PY) -m pytest -q -m "not slow" --cov=otgan --cov-report=term-missing

test-all:
	$(PY) -m pytest -q --cov=otgan --cov-report=term-missing

smoke:
	otgan train -c configs/smoke.yaml --override fid_every=0 num_workers=0
	otgan sample -c configs/smoke.yaml --ckpt weights_smoke/ot_gan.pt -n 16 -o /tmp/smoke_samples.png

train:
	otgan train -c configs/mnist.yaml

ablate:
	otgan ablate -c configs/mnist.yaml --axis critic_sign

sample:
	otgan sample -c configs/mnist.yaml --ckpt weights/ot_gan.pt -n 64 -o samples.png

report:
	latexmk -pdf -cd report/OT_GAN_report.tex

export-weights:
	$(PY) scripts/export_generator.py
	shasum -a 256 dist/*.pt

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage dist build
