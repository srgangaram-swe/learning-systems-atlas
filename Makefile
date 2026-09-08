.PHONY: bootstrap format quality unit test integration check demo demo-sprint2 demo-sprint3 demo-sprint4 demo-sprint5 build

bootstrap:
	uv sync --locked --all-groups
	uv run pre-commit install

format:
	uv run ruff check --fix .
	uv run ruff format .

quality:
	uv lock --check
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src

unit:
	uv run pytest -m unit

test:
	uv run pytest --cov=learning_atlas --cov-report=term-missing

integration:
	uv run pytest -m integration

check: quality
	uv run pytest --cov=learning_atlas --cov-report=term-missing --cov-report=xml
	uv run python scripts/check_branch_coverage.py coverage.xml --minimum 90
	uv build

demo:
	uv run learning-atlas run-all --config-dir configs --output-dir runs/quickstart

demo-sprint2:
	uv run learning-atlas benchmark-supervised \
		--config-dir configs/supervised/sprint-02 \
		--output-dir runs/sprint-02

demo-sprint3:
	uv run learning-atlas benchmark-unsupervised \
		--config-dir configs/unsupervised/sprint-03 \
		--output-dir runs/sprint-03

demo-sprint4:
	uv run learning-atlas benchmark-deep \
		--config-dir configs/deep/sprint-04 \
		--output-dir runs/sprint-04

demo-sprint5:
	uv run learning-atlas benchmark-reinforcement \
		configs/reinforcement/sprint-05/reinforcement.yaml \
		--output-dir runs/sprint-05

build:
	uv build

.PHONY: docs notebooks reference-parity
docs:
	uv run python scripts/build_api_reference.py --check
	uv run mkdocs build --strict

notebooks:
	uv run python scripts/execute_notebooks.py

reference-parity:
	uv run python scripts/reference_parity.py runs/reference-parity
