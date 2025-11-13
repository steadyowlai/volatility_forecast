.PHONY: help venv install clean test

help: ## List available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv: ## Create a local virtual environment at .venv
	python3 -m venv .venv

install: venv ## Install base dependencies into .venv
	. .venv/bin/activate; pip install -U pip && pip install -r requirements/base.txt

test: ## Run tests (none yet; placeholder for later)
	. .venv/bin/activate; pytest -q || true

clean: ## Remove caches and temporary files
	rm -rf __pycache__ .pytest_cache .venv