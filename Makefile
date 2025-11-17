.PHONY: help build build-base build-ingest run-ingest ingest clean clean-data venv install test

help: ## List available commands
	@echo "Volatility Forecast - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## ' Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Docker Build Commands
build: build-base build-ingest ## Build all Docker images

build-base: ## Build base Docker image
	docker build -t vf-base -f Dockerfile.base .

build-ingest: ## Build ingest service image
	docker build -t vf-ingest -f services/ingest/Dockerfile .

# Docker Run Commands
run-ingest: ## Run ingest service (downloads market data)
	docker-compose up ingest

ingest: build-ingest run-ingest ## Build and run ingest service

# Data Management
verify-data: ## Verify ingested data
	@echo "Raw market partitions:"
	@ls data/raw.market 2>/dev/null | wc -l || echo "0"
	@echo "Curated market partitions:"
	@ls data/curated.market 2>/dev/null | wc -l || echo "0"
	@echo ""
	@echo "Date range:"
	@ls data/raw.market 2>/dev/null | head -1 || echo "No data"
	@echo "to"
	@ls data/raw.market 2>/dev/null | tail -1 || echo "No data"

clean-data: ## Remove all downloaded data (use with caution!)
	rm -rf data/raw.market data/curated.market

# Local Development
venv: ## Create a local virtual environment at .venv
	python3 -m venv .venv

install: venv ## Install base dependencies into .venv
	. .venv/bin/activate; pip install -U pip && pip install -r requirements/base.txt

test: ## Run tests (placeholder for later)
	@echo "Tests not implemented yet"

# Cleanup
clean: ## Remove caches and temporary files
	rm -rf __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean clean-data ## Remove everything (cache, data, venv)
	rm -rf .venv