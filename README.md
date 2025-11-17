# Volatility Forecast System

ML-powered system to forecast 5-day realized volatility (RV_5d) for SPY with explainability.

## Quick Start

```bash
# Using Makefile (recommended)
make build        # Build all Docker images
make run-ingest   # Run ingest service (downloads 10 years of market data)
make verify-data  # Verify downloaded data

# Or use one command
make ingest       # Build and run ingest service

# View all available commands
make help
```

### Alternative (Docker commands)

```bash
# 1. Build images
docker build -t vf-base -f Dockerfile.base .
docker build -t vf-ingest -f services/ingest/Dockerfile .

# 2. Run ingest service
docker-compose up ingest
```

## Configuration

Edit `.env` to customize:
- `LOOKBACK_PERIOD`: Data history to download (default: `10y`)
  - Valid: `1y`, `2y`, `5y`, `10y`, `15y`, `max`

## Project Structure

```
├── services/          # Microservices
│   └── ingest/       # Market data ingestion
├── libs/             # Shared schemas and utilities
├── data/             # Data lake (partitioned Parquet)
│   ├── raw.market/
│   └── curated.market/
├── local/            # Private documentation (gitignored)
├── .env              # Configuration (gitignored)
└── docker-compose.yml
```

## Data Assets

- **SPY**: S&P 500 ETF
- **^VIX**: CBOE Volatility Index (30-day)
- **^VIX3M**: CBOE 3-Month Volatility
- **TLT**: 20+ Year Treasury Bond ETF
- **HYG**: High Yield Corporate Bond ETF

## Data Output

Data is partitioned by date in Parquet format:
- `data/raw.market/date=YYYY-MM-DD/` - OHLCV data
- `data/curated.market/date=YYYY-MM-DD/` - Daily returns

## Development

See `local/commands.txt` for detailed Docker commands.

## Architecture

Level 1 (Current): Local Docker-based pipeline
- Ingest → Features → Train → Infer → API → Dashboard

Level 2+: AWS deployment with SageMaker, Step Functions, etc.
