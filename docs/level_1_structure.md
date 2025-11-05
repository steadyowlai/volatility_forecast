# 📁 Project Folder Structure — Level 1 (Local Dockerized ML System)

This document describes the standard repository layout for the **ml-risk** project.  
It is organized as a modular, production-ready monorepo: one service per component, shared libraries under `libs/`, and containerized orchestration with Docker Compose.

---

## 🧱 Top-Level Layout

ml-risk/
├── README.md
├── LEVEL1_SCOPE.md
├── LEVEL1_STRUCTURE.md
├── Makefile
├── docker-compose.yml
├── Dockerfile.base
├── requirements/
│   ├── base.in
│   ├── base.txt
│   ├── ingest.txt
│   ├── features.txt
│   ├── train.txt
│   ├── infer.txt
│   ├── api.txt
│   └── dashboard.txt
├── config/
│   ├── features.yaml
│   └── env.example
├── libs/
│   ├── init.py
│   ├── io.py
│   ├── schemas.py
│   ├── features_core.py
│   ├── metrics.py
│   └── timecv.py
├── services/
│   ├── ingest/
│   │   ├── Dockerfile
│   │   └── app.py
│   ├── features/
│   │   ├── Dockerfile
│   │   └── app.py
│   ├── train/
│   │   ├── Dockerfile
│   │   └── app.py
│   ├── infer/
│   │   ├── Dockerfile
│   │   └── app.py
│   ├── api/
│   │   ├── Dockerfile
│   │   └── main.py
│   └── dashboard/
│       ├── Dockerfile
│       └── app.py
├── data/
│   ├── raw.market/
│   ├── curated.market_daily/
│   ├── features_daily/
│   ├── labels_daily/
│   └── predictions_daily/
├── artifacts/
│   ├── model/
│   └── shap/
├── tests/
│   ├── test_schemas.py
│   ├── test_features.py
│   ├── test_labels_alignment.py
│   ├── test_cv_split.py
│   └── test_metrics_gate.py
└── .github/
└── workflows/
└── ci.yml

---

## 📦 Directory Details

| Path | Description |
|------|--------------|
| **README.md** | Quick-start instructions, run commands, overview. |
| **LEVEL1_SCOPE.md** | Functional + non-functional requirements and feature list. |
| **LEVEL1_STRUCTURE.md** | This document — reference for repo layout. |
| **Makefile** | One-liners to run pipeline components (`make ingest`, `make features`, etc.). |
| **docker-compose.yml** | Orchestration of all Level 1 services (local runs). |
| **Dockerfile.base** | Shared base image for all service Dockerfiles (Python 3.11 + deps). |

### requirements/
Dependency management.  
Each service has its own requirement file extending `base.txt` so images stay small and isolated.

### config/
Configuration files and environment variables.  
`features.yaml` toggles optional features (e.g., enable cross-asset correlations).  
`env.example` lists local variables (AWS keys, bucket name placeholders).

### libs/
Shared Python package for reusable logic and data contracts.

| File | Responsibility |
|------|----------------|
| **io.py** | Read/write Parquet partitions (local or S3 path-compatible). |
| **schemas.py** | Pandera DataFrameSchemas for all datasets (contracts). |
| **features_core.py** | RSI, realized volatility, drawdown, correlations, etc. |
| **metrics.py** | RMSE, QLIKE, baseline comparison. |
| **timecv.py** | Walk-forward cross-validation utilities. |

### services/
Each subfolder = independent micro-service with its own Dockerfile and code entry point.

| Service | Purpose | Output / API |
|----------|----------|--------------|
| **ingest/** | Fetch SPY, VIX, VIX3M, TLT, HYG via yfinance. | `raw.market/`, `curated.market_daily/` |
| **features/** | Compute engineered Level 1 features. | `features_daily/` |
| **train/** | Train XGBoost model, evaluate, save artifact. | `artifacts/model/` |
| **infer/** | Run batch inference using latest model. | `predictions_daily/` |
| **api/** | Serve `/predict` endpoint via FastAPI. | JSON predictions |
| **dashboard/** | Visualize results via Streamlit. | Web UI (port 8501) |

### data/
Local Parquet datasets (mirrors S3 layout used in later levels).  
All files are append-only and validated by schemas.

### artifacts/
Model artifacts and SHAP explanations (ignored by Git).  
`model/latest.joblib` is loaded by infer/api services.

### tests/
Unit and integration tests validating features, schemas, CV, and metrics.  
Executed locally or in CI.

### .github/workflows/
CI configuration for linting, testing, and Docker build checks (GitHub Actions).

---

## ⚙️ Execution Flow
ingest → features → train → infer → api → dashboard


- **ingest:** downloads & curates raw data  
- **features:** builds leak-free features  
- **train:** fits and validates model  
- **infer:** generates latest prediction  
- **api:** serves results via HTTP  
- **dashboard:** displays charts locally

---

## 🚀 Run Commands (Make Targets)

| Command | Action |
|----------|--------|
| `make build` | Build all Docker images. |
| `make ingest` | Run data ingestion. |
| `make features` | Generate features. |
| `make train` | Train model. |
| `make infer` | Produce predictions. |
| `make api` | Launch FastAPI (port 8000). |
| `make dash` | Launch Streamlit (port 8501). |
| `make all` | Full pipeline: ingest → features → train → infer. |

---

## ✅ Notes

- Keep each service’s Dockerfile minimal; inherit from `Dockerfile.base`.  
- Validate every dataset against its schema before writing.  
- Never commit data or artifacts; they remain in `.gitignore`.  
- The same structure will scale to Level 2 (AWS ECS) and Level 3 (SageMaker Pipelines).

---