poetry-check:
	@echo "🔒 Verifying that poetry.lock is consistent with pyproject.toml..."
	poetry lock --check

format:
	isort .
	black .

format-check:
	@echo "⚫ Checking code format..."
	isort --check-only --diff .
	black --check .

type-check:
	@echo "👮 Running type checker"
	mypy .

static-checks: poetry-check format-check type-check

test:
	@echo "🏃 Running tests..."
	pytest .

all-checks: static-checks test
	@echo "✅ Great success!"

clean:
	rm -rf dist

build:
	poetry build

build-docs:
	mkdocs build

install-dev:
	poetry install --all-groups