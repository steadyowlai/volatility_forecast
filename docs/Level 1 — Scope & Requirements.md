# **Level 1 — Scope & Requirements (Markdown)**

## **1\) Summary**

Build a **Dockerized, end-to-end ML pipeline** that forecasts **5-day realized volatility (RV\_5d)** for **SPY**, with explainable outputs, a local **FastAPI** /predict endpoint, and a simple **dashboard**. Data → features → model → batch predictions → API → dashboard all run locally (or with S3 for storage) using **docker compose**.

This level establishes **stable data contracts** and **APIs** that Levels 2–3 will reuse.

---

## **2\) Objectives**

* Predict **RV\_5d** for SPY daily (EOD data).

* Produce **SHAP** top drivers for interpretability.

* Expose a **local API** (/predict) and a **dashboard** for visualization.

* Write **Parquet** datasets to **S3** (or local S3-compatible store) following stable schemas used in higher levels.

---

## **3\) In Scope**

* **Assets:** SPY, VIX, VIX3M, TLT, HYG (daily)

* **Features (core, leak-free):** returns, realized vol, drawdown, RSI, VIX & term structure, optional stock-bond/credit correlations

* **Modeling:** XGBoost/LightGBM regression for rv\_5d

* **Explainability:** SHAP (top contributors per prediction)

* **Outputs:** predictions\_daily Parquet \+ local API \+ dashboard

* **Tooling:** Docker, docker compose, Makefile, basic tests

### **Out of Scope (for Level 1\)**

* Macro features and regime discovery (GMM/HMM)

* RAG/LLM components

* Cloud orchestration/CI/CD (moved to Level 2\)

* Real-time hosting (Level 2/3)

---

## **4\) Data Sources**

* **Market:** public daily bars via yfinance (or equivalent) for SPY, VIX, VIX3M, TLT, HYG

* **No macro** at Level 1 (schema placeholders allowed but unpopulated)

## **5\) Data Contracts (S3 Layout, Parquet)**

s3://\<bucket\>/  
  raw.market/                 (symbol, date, open, high, low, close, volume)  
  curated.market\_daily/       (symbol, date, ohlc, ret, adj\_close)

  features\_daily/             (date, \<feature columns...\>)  
  labels\_daily/               (date, rv\_5d)

  predictions\_daily/          (date, rv\_5d\_hat, top\_drivers\[\])

* Partition by date (e.g., date=YYYY-MM-DD)

* **Immutability:** append-only; no in-place edits

* **Schema versioning:** add schema\_version column if desired

## **6\) Feature Set (Level 1\)**

**Target:**

* rv\_5d \= sqrt( sum\_{i=1..5} r\_{t+i}^2 ) using daily log returns

**Features (≈15–18 total):**

* **Returns:** spy\_ret\_1d, spy\_ret\_5d, spy\_ret\_20d

* **Realized Vol:** spy\_vol\_5d, spy\_vol\_10d, spy\_vol\_20d

* **Drawdown:** drawdown\_60d

* **Vol Indices:** vix, vix3m, vix\_term \= vix3m / vix

* **Momentum:** rsi\_spy\_14

* **Cross-Asset (optional at L1, nice-to-have):** corr\_spy\_tlt\_20d, corr\_spy\_hyg\_20d, hyg\_tlt\_spread

**Leakage guardrails:**

* All features computed **as-of close t**

* No future info; rolling windows anchored at or before t

---

## **7\) Modeling & Evaluation**

* **Model:** XGBoost or LightGBM regressor

* **CV:** walk-forward (expanding window)

* **Metrics:** RMSE, QLIKE

* **Baseline:** 20-day historical RV; require non-trivial improvement

* **Explainability:** SHAP values; log top 3 drivers per prediction

---

## **8\) Interfaces & APIs**

### **REST (local FastAPI)**

POST /predict  
Request:  
{  
  "as\_of\_date": "YYYY-MM-DD",  
  "feature\_vector": { "spy\_vol\_10d": 0.12, "vix": 19.4, "vix\_term": 1.12, "...": "..." }  
}  
Response:  
{  
  "rv\_5d\_hat": 0.226,  
  "regime\_probs": null,  
  "top\_drivers": \["vix\_term","spy\_vol\_10d","rsi\_spy\_14"\]  
}

* This contract is **stable** across all levels.

### **Batch Output**

* predictions\_daily Parquet with columns: date, rv\_5d\_hat, top\_drivers\[\]

---

## **9\) Local Architecture (Docker-first)**

**Services (each with own Dockerfile):**

* services/ingest: fetch market data → raw.market/, curated.market\_daily/

* services/features: compute L1 features → features\_daily/

* services/train: train model → artifacts/model/ in S3

* services/infer: batch inference → predictions\_daily/

* services/api: FastAPI for /predict (reads model from S3/artifacts)

* services/dashboard: Streamlit (reads predictions\_daily/)

**Orchestration:** docker compose runs ingest → features → train → infer; api & dashboard are long-lived.

ml-risk/  
  services/{ingest,features,train,infer,api,dashboard}/  
  libs/                       \# shared IO, utils, schema  
  tests/  
  docker-compose.yml  
  Makefile  
  README.md

**Configuration:**

* .env for AWS creds & bucket name

* config/\*.yaml for feature flags (e.g., enable\_optional\_correlations)

---

## **10\) Non-Functional Requirements (Level 1 targets)**

* **Reproducibility:** deterministic runs via Docker; fixed random seeds

* **Runtime:** end-to-end local run \< 10 min on laptop hardware

* **Observability:** structured logs (JSON) to stdout; simple run summary

* **Cost:** $0–low using free data \+ S3 (minimal storage)

---

## **11\) Testing & Quality**

* **Unit tests:** feature formulas, schema validation, leakage checks

* **Integration tests:** ingest→features→train happy-path

* **Performance gate:** fail training if RMSE/QLIKE not ≥ **1–2%** better than baseline on OOS slice

* **Data validation:** basic pandera/pydantic schema checks

---

## **12\) Acceptance Criteria**

* One-command local run (docker compose up) completes: **ingest→features→train→infer**

* predictions\_daily Parquet produced with expected schema and recent date

* /predict endpoint runs locally and returns SHAP top\_drivers

* Dashboard shows latest prediction & drivers

* Model beats naive 20-day RV baseline by **≥ 1–2%** on OOS (configurable gate)

---

## **13\) Nice-to-Haves (still Level 1\)**

* Store SHAP values per date in a side table

* Basic EDA notebook (read-only) using the same Parquet outputs

* Local MinIO as S3-compatible storage if you don’t want to hit AWS yet

---

## **14\) Resume Bullets (Level 1\)**

* **Containerized** an end-to-end ML pipeline to forecast **5-day market volatility** with **SHAP explainability**, producing **Parquet datasets** and exposing a **FastAPI** /predict endpoint and dashboard.

* Implemented **walk-forward validation** with **RMSE/QLIKE** gates, achieving **measurable lift vs a 20-day RV baseline**, and established **stable data/API contracts** for seamless cloud promotion.

