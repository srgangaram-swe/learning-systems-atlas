.PHONY: bootstrap format quality unit test integration check demo build

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
	uv build

demo:
	uv run learning-atlas run-all --config-dir configs --output-dir runs/quickstart

build:
	uv build
