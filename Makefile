PYTHON ?= python3
CONDA ?= conda
LIBERO_ENV_NAME ?= agent-for-robot-libero

.PHONY: bootstrap gateway-test source-check strict-runtime-test smoke verify-publication

bootstrap:
	CONDA=$(CONDA) LIBERO_ENV_NAME=$(LIBERO_ENV_NAME) ./scripts/bootstrap.sh

gateway-test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=libero/src $(CONDA) run -n $(LIBERO_ENV_NAME) python -m unittest discover -s libero/tests -p 'test_*.py' -v

strict-runtime-test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=libero/runtime/strict $(PYTHON) -m unittest libero/runtime/strict/test_strict_projection.py -v

source-check:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/validate_release.py

smoke: source-check strict-runtime-test verify-publication

verify-publication:
	./scripts/verify_publication.sh
