\# Project Requirements: Macro-Aware Market Risk & Regime Forecasting  
\*\*Scope:\*\* Production-grade ML system on AWS that (1) forecasts short-term market \*\*risk\*\* (5-day realized volatility) and (2) estimates \*\*regime probabilities\*\* (calm / transition / risk-off), with (3) \*\*unsupervised regime discovery\*\* and (4) a \*\*RAG\*\* layer for explainable, cited narratives. Macro variables are \*inputs only\*.

\---

\#\# 1\) Objectives & Success Criteria  
\#\#\# Functional  
\- Predict \*\*RV\_5d\*\* for a benchmark (e.g., SPY) and produce \*\*regime probabilities\*\* daily before market open.  
\- Discover latent regimes via \*\*unsupervised clustering\*\*; expose cluster labels/probabilities as features and analytics.  
\- Serve \*\*batch outputs\*\* (S3, DynamoDB) and an optional \*\*real-time endpoint\*\* (API Gateway → SageMaker).  
\- Provide \*\*RAG Q\&A\*\* and a \*\*daily risk brief\*\* with \*\*citations\*\* (FOMC statements, CPI/NFP releases, etc.).

\#\#\# Non-Functional (targets)  
\- \*\*Availability:\*\* ≥ 99% for daily pipeline; \*\*SLO:\*\* predictions available by 08:00 local exchange time.  
\- \*\*Latency:\*\* batch inference \< 5 min; RAG question answer \< 2.5 s p95.  
\- \*\*Cost:\*\* ≤ $X/month at MVP scale (SageMaker serverless/batch preferred).  
\- \*\*MLOps:\*\* reproducible, versioned, automated retraining; full lineage.

\#\#\# Acceptance  
\- Out-of-sample improvement vs baseline (e.g., 20-day historical RV): \*\*RMSE ↓ ≥ 5%\*\* or \*\*QLIKE ↓ ≥ 5%\*\*.  
\- Regime classifier: \*\*Brier ↓ vs persistence\*\* and \*\*AUC ≥ 0.65\*\* (MVP).  
\- RAG: \*\*Recall@5 ≥ 0.8\*\* on curated Q/A set; answers include ≥ 2 citations.

\---

\#\# 2\) Data & Schemas  
\#\#\# Sources (daily)  
\- \*\*Market:\*\* SPY, sector ETFs, VIX/VIX3M, TLT/HYG, DXY, GLD, WTI; optional: 2y/10y yields, credit spreads.  
\- \*\*Macro (contextual only):\*\* CPI YoY, Fed Funds (or 3m), 10y–2y slope, ISM PMI, Unemployment.  
\- \*\*Docs for RAG:\*\* FOMC statements/minutes, BLS/BEA releases (CPI/NFP), optional earnings transcripts.

\#\#\# Storage (S3, Parquet, partitioned)  
\- \`raw.market/\` (symbol, date, ohlcv)  
\- \`raw.macro/\` (series\_id, macro\_value\_date, value, release\_datetime)  
\- \`curated.market\_daily/\` (symbol, date, ohlc, ret, adj\_close)  
\- \`curated.macro\_release\_aligned/\` (date, {cpi\_yoy, ffr, slope\_10y\_2y, ism\_pmi, unemp}, as\_of\_release)  
\- \`features\_daily/\` (date, …features…)  
\- \`labels\_daily/\` (date, rv\_5d, regime\_5d)  
\- \`predictions\_daily/\` (date, rv\_5d\_hat, regime\_probs, top\_drivers\[\])  
\- \`docs.raw/\` (source, url, fetched\_at, blob)  
\- \`docs.curated/\` (doc\_id, published\_at, source, text\_md)  
\- \`docs.chunks/\` (chunk\_id, doc\_id, text\_chunk, embedding, metadata)

\#\#\# Feature Table (examples; all leak-free, computed as-of close t)  
\- Market: \`spy\_ret\_{1,5,10,20}d\`, \`spy\_vol\_{5,10,20}d\`, \`drawdown\_60d\`, \`vix\`, \`vix3m\`, \`vix\_term=vix3m/vix\`,  
  \`corr\_spy\_tlt\_20d\`, \`corr\_spy\_hyg\_20d\`, \`hyg\_tlt\_spread\`, \`rsi\_spy\_14\`, \`skew\_20d\`, \`kurt\_20d\`  
\- Macro (z-scores; forward-filled \*\*day after release\*\*): \`cpi\_yoy\_z\`, \`ffr\_real\`, \`slope\_10y\_2y\`, \`ism\_pmi\_z\`, \`unemp\_rate\_z\`  
\- Unsupervised features (added post clustering): \`regime\_cluster\_id\`, \`prob\_cluster\_{0..K-1}\`, \`dist\_to\_centroid\_{k}\`

\#\#\# Targets  
\- \`rv\_5d \= sqrt(sum\_{i=1..5} r\_{t+i}^2)\`  
\- \`regime\_5d\` (binary or 3-class): threshold RV quantiles or mapped from clusters.

\---

\#\# 3\) Modeling  
\#\#\# Supervised  
\- \*\*Regression (primary):\*\* XGBoost / LightGBM → predict \`rv\_5d\`  
  \- Loss/metrics: RMSE, \*\*QLIKE\*\*; walk-forward CV, expanding window.  
\- \*\*Classification (optional):\*\* XGBoostClassifier → predict \`regime\_5d\`  
  \- Metrics: AUC, \*\*Brier score\*\*, calibration curves.

\#\#\# Unsupervised (daily refresh or weekly)  
\- \*\*GMM\*\* (preferred) or \*\*KMeans/MiniBatchKMeans\*\* on standardized regime features:  
  \- Inputs: \`{spy\_vol\_10d, vix, vix\_term, corr\_spy\_tlt\_20d, hyg\_tlt\_spread, drawdown\_60d, …}\`  
  \- \`K \= 3–5\`. Save \`cluster\_id\`, \`proba\_k\`.  
\- Optional temporal smoothing with \*\*HMM\*\* over \`cluster\_id\` sequence.

\#\#\# Explainability  
\- \*\*SHAP\*\* on supervised models; log top contributors per prediction.

\---

\#\# 4\) RAG Component  
\#\#\# Pipeline  
\- \*\*Ingest\*\* (Lambda/Glue): fetch & parse FOMC/BLS/BEA pages; store in \`docs.raw/\`.  
\- \*\*Curate & Chunk\*\* (SageMaker Processing): HTML→Markdown, chunk \~800–1,000 tokens with 15% overlap; metadata: \`published\_at\`, \`source\`, \`doc\_type\`, \`asset\_tags\`.  
\- \*\*Embed\*\*: Bedrock (e.g., Titan Embeddings) → store vectors in \*\*OpenSearch Serverless\*\* (k-NN) or \*\*Aurora pgvector\*\*.  
\- \*\*Retrieve\*\*: filter by recency/source → top-k by cosine; optional local reranker from JumpStart.  
\- \*\*Generate\*\*: Bedrock LLM (e.g., Claude/Llama via Bedrock) with structured prompt including today’s \`predictions\_daily\` JSON (rv\_5d\_hat, regime\_probs, top\_drivers). Return citations.

\#\#\# RAG API  
\- \`POST /qa\` → \`{question, constraints?} → {answer, citations\[\]}\`  
\- \`POST /daily-brief\` → \`{summary\_md, bullets\[\], citations\[\]}\`

\---

\#\# 5\) AWS Architecture

EventBridge (schedules)

└──\> Step Functions (or SageMaker Pipelines)  ───────────────────────────────────────────────────────────────────────────────┐

├─ ETL: Lambda/Glue → S3 raw/curated → Glue Catalog/Athena                                                            │

├─ Feature build: SM Processing → S3 features\_daily                                                                   │

├─ Labels build: SM Processing → S3 labels\_daily                                                                      │

├─ Unsupervised clustering: SM Processing → S3 cluster\_features                                                        │

├─ Train/eval: SM Training (XGBoost/LightGBM) → SM Model Registry                                                      │

├─ Conditional deploy: (metrics gate) → SM Hosting (real-time) or Batch Transform → S3 predictions / Dynamo            │

└─ Monitoring: SM Model Monitor / custom jobs → CloudWatch / QuickSight                                                │

RAG sidecar:

Docs ingest (Lambda/Glue) → S3 docs.raw → Curate/Chunk (SM Processing) → Embeddings (Bedrock)

→ Vector DB (OpenSearch/pgvector) → FastAPI (ECS Fargate or SM Endpoint) behind API Gateway

\*\*Security & Ops:\*\* IAM least-privilege; KMS encryption for S3/OpenSearch; Secrets Manager for API keys; VPC endpoints for SageMaker; CloudWatch alarms.

\---

\#\# 6\) Orchestration & Schedules  
\- \*\*Market ETL:\*\* daily 22:00 UTC (post close).    
\- \*\*Macro ETL:\*\* at official release times (CPI/NFP 13:30 UTC), forward-fill from \*\*next trading day\*\*.    
\- \*\*Feature/Label Build:\*\* after ETL completion.    
\- \*\*Unsupervised Fit:\*\* nightly or weekly (configurable).    
\- \*\*Training & Eval:\*\* nightly (expand window) or \*\*on drift\*\*.    
\- \*\*Inference:\*\* nightly batch; optional intra-day refresh.    
\- \*\*RAG Index Refresh:\*\* daily; \`daily-brief\` by 07:30 local.

\---

\#\# 7\) Promotion, Monitoring, Drift  
\- \*\*Promotion rules:\*\* new model registered if \`QLIKE\` (rv) and \*\*Brier\*\* (regime) improve by ≥ threshold with paired t-test or min Δ.  
\- \*\*Shadow deploy:\*\* 2–4 weeks optional; compare live metrics before traffic shift.  
\- \*\*Data quality:\*\* Great Expectations; stop-the-line on failed contracts (missing symbols, stale macro).  
\- \*\*Drift:\*\* PSI/KS on key features and targets; alert thresholds (e.g., PSI \> 0.2).  
\- \*\*Post-deploy metrics:\*\* rolling RMSE/QLIKE, calibration; written to CloudWatch \+ QuickSight.

\---

\#\# 8\) APIs & Contracts  
\#\#\# Prediction (real-time optional)  
\`POST /predict\`  
\`\`\`json  
{  
  "as\_of\_date": "2025-10-24",  
  "feature\_vector": { "spy\_vol\_10d": 0.12, "vix": 19.4, "vix\_term": 1.12, "cpi\_yoy\_z": 0.8, "...": "..." }  
}

{  
  "rv\_5d\_hat": 0.226,  
  "regime\_probs": {"calm": 0.21, "transition": 0.17, "risk\_off": 0.62},  
  "top\_drivers": \["vix\_term","spy\_vol\_10d","corr\_spy\_tlt\_20d"\]  
}

### **Batch outputs (S3 Parquet \+ DynamoDB)**

predictions\_daily\_{YYYYMMDD}.parquet with columns: date, rv\_5d\_hat, regime\_probs.\*, top\_drivers\[\]

### **RAG (see Section 4\)**

---

## **9\) CI/CD & Reproducibility**

* **Repos:** infra/ (Terraform/CFN), pipelines/, ml/, rag/, app/.

* **Pipelines:** CodePipeline → build Docker images (ECR), run unit/integration tests, deploy Step Functions/SageMaker Pipeline, update endpoints.

* **Experiment tracking:** SageMaker Experiments or MLflow on EC2.

* **Data/versioning:** S3 object versioning; model registry versions; artifact hashes.

---

## **11\) Deliverables**

* **Code** (IaC \+ pipelines \+ models \+ RAG \+ API).

* **Docs**: README with system diagram, data contracts, leakage tests, eval results.

* **Dashboards**: QuickSight/Streamlit—daily predictions, regime probs, SHAP, drift.

* **RAG demo**: Q\&A and daily brief with citations.

* **Resume bullets** (ready to paste).

---

## **12\) Risks & Mitigations**

* **Data leakage via macro releases** → enforce release-time lags; unit tests for “as-of” joins.

* **Overfitting** → walk-forward CV; conservative feature sets; simplify before deep models.

* **RAG hallucinations** → strict retrieval filters (recency, source), citations required; guardrails.

* **Cost bloat** → prefer batch/serverless; right-size instances; turn off idle endpoints.

---

## **13\) Resume Bullets (concise)**

* Built a **SageMaker**\-based pipeline to forecast **5-day market risk** and **regime probabilities**, with nightly retraining and blue/green deploys.

* Engineered **cross-asset** features and integrated **macro covariates** as slow contextual inputs; achieved **QLIKE ↓ ≥ 5%** vs baseline.

* Added **unsupervised regime discovery** (GMM/HMM) to learn latent states; used cluster probabilities as features to improve calibration.

* Implemented a **RAG explainability** layer (Bedrock \+ OpenSearch) to generate cited risk briefs and interactive Q\&A.

* Productionized with **S3/Glue/Athena**, **Model Registry**, **Model Monitor**, and **Great Expectations**; drift-triggered retrains and automated rollbacks.

