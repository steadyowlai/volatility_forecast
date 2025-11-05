# 🧱 Level 1 — Local Dockerized ML Pipeline

## 1️⃣ Goal

Build a **containerized, end-to-end ML system** that forecasts the **5-day realized volatility (RV₅d)** of the **S&P 500 ETF (SPY)**.  
The system should run entirely on a laptop using Docker Compose and output reproducible, explainable predictions.

---

## 2️⃣ Functional Requirements

| Area | Requirement |
|------|--------------|
| **Prediction** | Forecast `rv_5d = sqrt(sum(r_{t+1..t+5}^2))` once per trading day using daily SPY-based features. |
| **Explainability** | Compute SHAP values and expose top feature drivers for each prediction. |
| **APIs** | Serve predictions through a local **FastAPI** endpoint `/predict`. |
| **Dashboard** | Display latest forecast, SHAP drivers, and historical trend via **Streamlit**. |
| **Storage** | All datasets written to local Parquet partitions under `/data` (mirrors cloud S3 layout). |
| **Reproducibility** | Deterministic runs via Docker and fixed random seeds. |
| **Execution** | One command (`make all` or `docker compose up`) runs the full pipeline: `ingest → features → train → infer → api → dashboard`. |

---

## 3️⃣ Non-Functional Requirements

| Attribute | Target |
|------------|--------|
| **Runtime** | Full run < 10 min on laptop hardware. |
| **Reliability** | Schema-validated I/O (Pandera). |
| **Cost** | $0 (local, free data). |
| **Observability** | JSON logs to stdout + simple metrics summary. |
| **Extensibility** | Codebase compatible with Level 2 (AWS) and Level 3 (SageMaker). |

---

## 4️⃣ Core Features (Inputs)

| Feature | Category | Description |
|----------|-----------|-------------|
| `spy_ret_1d`, `spy_ret_5d`, `spy_ret_20d` | Returns | Log returns over 1, 5, 20 days. |
| `spy_vol_5d`, `spy_vol_10d`, `spy_vol_20d` | Realized Vol | Rolling √(Σ r²) over 5, 10, 20 days. |
| `drawdown_60d` | Drawdown | 1 − (close / 60-day max). |
| `vix`, `vix3m`, `vix_term = vix3m/vix` | Vol Index | Spot & 3-month implied vol; term structure. |
| `rsi_spy_14` | Momentum | 14-day RSI. |
| `corr_spy_tlt_20d` | Correlation | 20-day rolling corr (SPY, TLT). |
| `corr_spy_hyg_20d` | Correlation | 20-day rolling corr (SPY, HYG). |
| `hyg_tlt_spread` | Spread | `ret_HYG − ret_TLT`. |

All features computed **as-of close t** (no future info).

---

## 5️⃣ Data Contracts / Schemas

| Dataset | Columns | Partition Key | Produced By | Consumed By |
|----------|----------|---------------|--------------|-------------|
| `raw.market/` | `symbol,date,open,high,low,close,volume` | `date` | ingest | features |
| `curated.market_daily/` | `symbol,date,ret,adj_close` | `date` | ingest | features |
| `features_daily/` | `date,<features…>` | `date` | features | train + infer |
| `labels_daily/` | `date,rv_5d` | `date` | train | train (eval) |
| `predictions_daily/` | `date,rv_5d_hat,top_drivers[]` | `date` | infer | api + dashboard |

All files are **append-only** and **validated** by Pandera tests.

---

## 6️⃣ Modeling

- **Algorithm:** XGBoost / LightGBM regression  
- **CV:** Walk-forward expanding-window (time-aware)  
- **Metrics:** RMSE, QLIKE  
- **Baseline:** 20-day realized vol (`spy_vol_20d`)  
- **Acceptance Gate:** Out-of-sample RMSE or QLIKE ≥ 1–2 % better than baseline  
- **Explainability:** SHAP (top 3 drivers logged per prediction)

---

## 7️⃣ Interfaces

### REST API (`/predict`)
**Request**
```json
{
  "as_of_date": "YYYY-MM-DD",
  "feature_vector": { "spy_vol_10d": 0.12, "vix": 19.4, "vix_term": 1.12, "...": "..." }
}

Response
{
  "rv_5d_hat": 0.226,
  "regime_probs": null,
  "top_drivers": ["vix_term","spy_vol_10d","rsi_spy_14"]
}


Dashboard

Simple Streamlit UI showing:
	•	latest RV₅d forecast
	•	SHAP top drivers
	•	time-series chart of predictions

⸻

8️⃣ Directory & Execution Overview

ml-risk/
  libs/                 # shared utils + schemas + metrics
  services/
    ingest/             # fetch data
    features/           # compute features
    train/              # train model
    infer/              # batch predict
    api/                # FastAPI /predict
    dashboard/          # Streamlit UI
  data/                 # Parquet outputs (gitignored)
  artifacts/            # model artifacts, SHAP
  docker-compose.yml
  Makefile


9️⃣ Acceptance Criteria
	•	One-command run completes: ingest → features → train → infer
	•	Parquet partitions exist and validate against schemas.
	•	Model beats baseline by ≥ 1–2 %.
	•	/predict endpoint returns valid JSON response.
	•	Dashboard visualizes latest prediction.

⸻

🔟 Next Level Preview

Level 2 will move this exact pipeline to AWS (ECS Fargate + API Gateway) using the same code, schemas, and data contracts.
Level 3 will extend features with macro & regime inputs and add drift-aware retraining.

⸻

