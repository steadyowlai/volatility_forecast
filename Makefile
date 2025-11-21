.PHONY: help build build-base build-ingest run-ingest ingest clean clean-data venv install test

help: ## List available commands
	@echo "Volatility Forecast - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## ' Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Docker Build Commands
build: ## Build all Docker images
	docker-compose build

build-base: ## Build base Docker image
	docker-compose build base

build-ingest: ## Build ingest service image
	docker-compose build ingest

build-features: ## Build features service image
	docker-compose build features

# Docker Run Commands
run-ingest: ## Run ingest service (downloads market data)
	docker-compose up ingest

ingest: ## Build and run ingest service
	docker-compose up --build ingest

run-features: ## Run features service (computes 21 Level 1 features)
	docker-compose up features

features: ## Build and run features service
	docker-compose up --build features

# Pipeline Commands - run multiple services in sequence
pipeline: ## Run full pipeline: ingest → features
	@echo "Running data pipeline..."
	@echo "Step 1/2: Ingesting market data..."
	docker-compose up ingest
	@echo "Step 2/2: Computing features..."
	docker-compose up features
	@echo "Pipeline complete!"

# Data Management
verify-data: ## Verify ingested data and features
	@echo "📊 Data Pipeline Status"
	@echo "======================="
	@echo ""
	@echo "Raw market partitions:"
	@ls data/raw.market 2>/dev/null | wc -l || echo "0"
	@echo ""
	@echo "Curated market partitions:"
	@ls data/curated.market 2>/dev/null | wc -l || echo "0"
	@echo ""
	@echo "Feature partitions:"
	@ls data/features.L1 2>/dev/null | wc -l || echo "0"
	@echo ""
	@echo "Date range (raw data):"
	@ls data/raw.market 2>/dev/null | head -1 || echo "No data"
	@echo "to"
	@ls data/raw.market 2>/dev/null | tail -1 || echo "No data"

clean-data: ## Remove all downloaded data (use with caution!)
	rm -rf data/raw.market data/curated.market data/features.L1

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