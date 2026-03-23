uv-lock-check:
	@echo "🔒 Verifying that uv.lock is consistent with pyproject.toml..."
	uv lock --check

format:
	uv run isort .
	uv run black .

format-check:
	@echo "⚫ Checking code format..."
	uv run isort --check-only --diff .
	uv run black --check .

type-check:
	@echo "👮 Running type checker"
	uv run mypy .

static-checks: uv-lock-check format-check type-check

test:
	@echo "🏃 Running tests..."
	uv run pytest .

all-checks: static-checks test
	@echo "✅ Great success!"

docs-serve:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build

docs-deploy:
	uv run mkdocs gh-deploy

clean:
	rm -rf dist

build:
	uv build
