# Volatility Forecast System

A production-grade ML system that forecasts 5-day realized volatility for SPY. Built with microservices architecture, containerized with Docker, and designed for deployment to AWS.

**Goal**: Predict how volatile the S&P 500 will be over the next 5 days, with full explainability via SHAP values.

## What's Built So Far

This is an evolving project. Here's what's working right now:

### Ingest Service
Downloads 10 years of market data from Yahoo Finance for SPY, VIX, VIX3M, TLT, and HYG. Data gets validated with Pandera schemas and saved as date-partitioned Parquet files. This gives us a clean data foundation to work with.

### Features Service  
Computes 21 features that capture market dynamics:
- **Returns**: SPY returns over 1d, 5d, 10d, 20d, 60d windows
- **Volatility**: Realized volatility using rolling windows
- **VIX Features**: Current VIX, VIX3M, and term structure (VIX3M/VIX ratio)
- **Technical**: 14-day RSI, 60-day drawdown
- **Cross-Asset**: Rolling correlations between SPY, TLT, and HYG
- **Spreads**: HYG-TLT spread, realized vol vs implied vol (VIX)

All features are stored in the same date-partitioned format for easy time-series work.

### 🔲 Coming Next
- Training service with MLflow model registry
- FastAPI prediction endpoint
- Streamlit dashboard for visualization
- AWS deployment with drift monitoring

## Quick Start

The easiest way to run this locally:

```bash
# See all available commands
make help

# Run the full data pipeline
make pipeline

# Or run services individually
make ingest      # Download market data
make features    # Compute features
```

If you prefer docker-compose directly:

```bash
docker-compose up ingest
docker-compose up features
```

## Configuration

Create a `.env` file (or use the default settings):

```bash
LOOKBACK_PERIOD=10y    # How much historical data to download
                        # Options: 1y, 2y, 5y, 10y, 15y, max
```

Ten years gives us enough data for solid model training without being overwhelming.

## Project Structure

```
├── services/
│   ├── ingest/          # Downloads market data from Yahoo Finance
│   └── features/        # Feature engineering (21 features)
├── libs/
│   └── schemas.py       # Pandera schemas for data validation
├── data/                # Local data lake (gitignored)
│   ├── raw.market/      # OHLCV data, partitioned by date
│   ├── curated.market/  # Daily returns and adjusted close
│   └── features.L1/     # Computed features
├── tests/               # pytest tests (unit + integration)
├── Makefile             # Convenient commands for running services
└── docker-compose.yml   # Service orchestration
```

## Data Schema

We track 5 assets that give us a view into equity, volatility, bonds, and credit markets:

| Symbol | Asset | Why It Matters |
|--------|-------|----------------|
| SPY | S&P 500 ETF | Our target (what we're predicting volatility for) |
| ^VIX | CBOE Volatility Index | Market's expectation of 30-day volatility |
| ^VIX3M | 3-Month VIX | Longer-term volatility expectations |
| TLT | 20+ Year Treasury ETF | Flight to safety indicator |
| HYG | High Yield Corporate Bonds | Credit risk and risk appetite |

Data flows through three layers:

1. **raw.market**: Raw OHLCV from Yahoo Finance
2. **curated.market**: Daily returns, adjusted close, cleaned up
3. **features.L1**: All computed features, ready for modeling

Everything is stored in date-partitioned Parquet format, which makes it easy to read specific time windows without loading the entire dataset.

## Architecture

Right now this runs entirely in Docker on your local machine. The architecture is designed so each service does one thing well:

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│   Ingest    │ ───> │   Features   │ ───> │   Training   │
│  (Yahoo)    │      │  (21 vars)   │      │  (XGBoost)   │
└─────────────┘      └──────────────┘      └──────────────┘
                                                    │
                                                    v
                                            ┌──────────────┐
                                            │   MLflow     │
                                            │  Registry    │
                                            └──────────────┘
```

Each service is stateless and communicates through the data layer (Parquet files). This makes it easy to test, debug, and eventually deploy to AWS where each service can run independently.

## Testing

Every function has pytest tests. Run them with:

```bash
make test              # Full test suite with coverage
make test-unit         # Just unit tests
make test-integration  # Just integration tests
```

We're targeting 80%+ code coverage to ensure things actually work before deploying to production.

## Development Workflow

The typical workflow when adding new features or services:

1. Write the core logic
2. Add Pandera schema for data validation
3. Write pytest tests (unit + integration)
4. Update docker-compose.yml
5. Add Makefile commands for convenience
6. Run the pipeline end to end to verify

## Future Plans

**Level 2**: Deploy to AWS with ECS, Lambda, S3, and RDS. Add CI/CD with GitHub Actions.

**Level 3**: Implement drift monitoring with Evidently AI, automated model retraining, and MLflow model registry with staging/production promotion.

This is being built as a portfolio project to demonstrate production ML engineering skills but the system is designed to actually work in a real production environment.

## Why This Tech Stack?

**Docker**: Makes the entire system reproducible. What works on my machine works everywhere.

**Parquet**: Columnar storage is fast and efficient for time-series data. Date partitioning means we can read exactly the data we need.

**Pandera**: Catches data quality issues early. If Yahoo Finance returns something weird, we know immediately.

**MLflow**: Industry standard for tracking experiments and managing model versions.

**pytest**: Writing tests as we build means fewer surprises later.

The stack is intentionally simple and focused on things that will actually run in production, not just in notebooks.

---

Built with Python 3.11, Docker, pandas, and a lot of coffee ☕
