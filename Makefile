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

build-train: ## Build training service image
	docker-compose build train

build-predict: ## Build prediction service image
	docker-compose build predict

build-validate: ## Build validation service image
	docker-compose build validate

# Docker Run Commands
run-ingest: ## Run ingest service (downloads market data)
	docker-compose up ingest

ingest: ## Build and run ingest service
	docker-compose up --build ingest

run-features: ## Run features service (computes 21 Level 1 features)
	docker-compose up features

features: ## Build and run features service
	docker-compose up --build features

run-train: ## Run training service (trains stacking ensemble)
	docker-compose up train

train: ## Build and run training service
	docker-compose up --build train

run-predict: ## Run prediction service (generates daily forecasts)
	docker-compose up predict

predict: ## Build and run prediction service
	docker-compose up --build predict

run-validate: ## Run validation service (validates predictions vs actuals)
	docker-compose up validate

validate: ## Build and run validation service
	docker-compose up --build validate

mlflow: ## Start MLflow tracking server
	docker-compose up -d mlflow
	@echo "MLflow UI available at http://localhost:5001"

mlflow-stop: ## Stop MLflow tracking server
	docker-compose stop mlflow

# Pipeline Commands - run multiple services in sequence
pipeline: ## Run full pipeline: ingest → features
	@echo "Running data pipeline..."
	@echo "Step 1/2: Ingesting market data..."
	docker-compose up ingest
	@echo "Step 2/2: Computing features..."
	docker-compose up features
	@echo "Pipeline complete!"

pipeline-train: ## Run full ML pipeline: ingest → features → train
	@echo "Running full ML pipeline..."
	@echo "Step 1/4: Starting MLflow server..."
	docker-compose up -d mlflow
	@sleep 3
	@echo "Step 2/4: Ingesting market data..."
	docker-compose up ingest
	@echo "Step 3/4: Computing features..."
	docker-compose up features
	@echo "Step 4/4: Training model..."
	docker-compose up train
	@echo "Pipeline complete! Check MLflow at http://localhost:5001"

pipeline-full: ## Run complete production pipeline: ingest → features → train → predict → validate
	@echo "Running complete production pipeline..."
	@echo "Step 1/6: Starting MLflow server..."
	docker-compose up -d mlflow
	@sleep 3
	@echo "Step 2/6: Ingesting market data..."
	docker-compose up ingest
	@echo "Step 3/6: Computing features..."
	docker-compose up features
	@echo "Step 4/6: Training model..."
	docker-compose up train
	@echo "Step 5/6: Generating predictions..."
	docker-compose up predict
	@echo "Step 6/6: Validating predictions..."
	docker-compose up validate
	@echo "Pipeline complete! Check MLflow at http://localhost:5001"

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
	@echo "Prediction partitions:"
	@ls data/predictions 2>/dev/null | wc -l || echo "0"
	@echo ""
	@echo "Date range (raw data):"
	@ls data/raw.market 2>/dev/null | head -1 || echo "No data"
	@echo "to"
	@ls data/raw.market 2>/dev/null | tail -1 || echo "No data"

clean-data: ## Remove all downloaded data (use with caution!)
	rm -rf data/raw.market data/curated.market data/features.L1 data/predictions

# Local Development
venv: ## Create a local virtual environment at .venv
	python3 -m venv .venv

install: venv ## Install base dependencies into .venv
	. .venv/bin/activate; pip install -U pip && pip install -r requirements/base.txt

test: ## Run all tests with coverage
	docker-compose up --build test

test-unit: ## Run only unit tests
	docker-compose run --rm test pytest tests/unit/ -v

test-integration: ## Run only integration tests
	docker-compose run --rm test pytest tests/integration/ -v

test-coverage: ## Run tests and generate HTML coverage report
	docker-compose run --rm test pytest tests/ -v --cov=services --cov=libs --cov-report=term-missing --cov-report=html

# Cleanup
clean: ## Remove caches and temporary files
	rm -rf __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean clean-data ## Remove everything (cache, data, venv)
	rm -rf .venv