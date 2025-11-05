# **Level 3 — Scope & Requirements (Markdown)**

## **1\) Summary**

Extend the Level 2 AWS pipeline with **advanced ML** and **enterprise MLOps**:

* Add **macro features** and **unsupervised regime discovery** (GMM/HMM smoothing).

* Migrate training to **SageMaker Pipelines** \+ **Model Registry** with **metrics-gated promotion**.

* Add **monitoring & drift** (SageMaker Model Monitor \+ data contracts).

* *(Optional)* Add a **RAG sidecar** for cited risk briefs & Q\&A.

   All **data/API contracts remain unchanged** (Level 1/2 compatible).

---

## **2\) Objectives**

* Improve **forecast calibration & accuracy** using macro context and regime features.

* Operationalize training with **SageMaker Pipelines** (experiments, lineage, registry).

* Enable **drift-aware retraining** and **automated rollbacks**.

* *(Optional)* Serve a **/daily-brief** and **/qa** endpoint with **citations**.

---

## **3\) In Scope**

* **Macro ingestion**: FRED (or equivalent) → raw.macro/ → curated.macro\_release\_aligned/ with release-time alignment.

* **Feature expansion**: Add macro z-scores & regime features to features\_daily/.

* **Unsupervised regimes**: GMM (K=3–5) on regime feature set; optional HMM smoothing.

* **SageMaker Pipelines**: Training, evaluation, registration, conditional deploy (batch/real-time).

* **Monitoring**: SageMaker Model Monitor, Great Expectations, feature/target drift (PSI/KS), alarms.

* **Promotion policy**: Metrics-gated with shadow/canary; version pin/rollback.

* **RAG (optional)**: Doc ingest → chunk → embeddings (OpenSearch/pgvector) → sidecar API.

### **Out of Scope**

* Strategy/PnL backtesting, order routing, or intraday tick modeling.

---

## **4\) Data Contracts (additive; backward compatible)**

s3://\<bucket\>/  
  raw.market/                      (symbol, date, ohlcv)  
  curated.market\_daily/            (symbol, date, ohlc, ret, adj\_close)

  raw.macro/                       (series\_id, macro\_value\_date, value, release\_datetime)  
  curated.macro\_release\_aligned/   (date, cpi\_yoy, ffr, slope\_10y\_2y, ism\_pmi, unemp, as\_of\_release)

  features\_daily/                  (date, spy\_ret\_\*, spy\_vol\_\*, rsi\_spy\_14, vix, vix3m, vix\_term,  
                                    corr\_spy\_tlt\_20d?, corr\_spy\_hyg\_20d?, hyg\_tlt\_spread?,  
                                    cpi\_yoy\_z, ffr\_real, slope\_10y\_2y, ism\_pmi\_z, unemp\_rate\_z,  
                                    regime\_cluster\_id, prob\_cluster\_0..K-1, dist\_to\_centroid\_k)  
  labels\_daily/                    (date, rv\_5d\[, regime\_5d?\])  
  predictions\_daily/               (date, rv\_5d\_hat, regime\_probs?, top\_drivers\[\])

  \# (Optional RAG)  
  docs.raw/                        (source, url, fetched\_at, blob)  
  docs.curated/                    (doc\_id, published\_at, source, text\_md)  
  docs.chunks/                     (chunk\_id, doc\_id, text\_chunk, embedding, metadata)

Existing consumers of features\_daily/ and /predict remain valid; new columns are additive.  
---

## **5\) Feature Set (additions)**

* **Macro (contextual, forward-filled from next trading day):**

  * cpi\_yoy\_z, ffr\_real, slope\_10y\_2y, ism\_pmi\_z, unemp\_rate\_z

* **Regime features (from unsupervised model):**

  * regime\_cluster\_id (int), prob\_cluster\_0..K-1, dist\_to\_centroid\_k

* **(Optional) Regime target/classifier**: regime\_5d (3-class: calm/transition/risk-off via RV quantiles or cluster mapping)

---

## **6\) Modeling**

* **Primary regression**: XGBoost/LightGBM → rv\_5d

  * Metrics: RMSE, QLIKE; walk-forward CV; expanding window.

* **Unsupervised**: GMM (K=3–5) on regime features; recompute nightly or weekly.

  * Optional HMM smoothing on cluster\_id sequence.

* **Optional classifier**: XGBoostClassifier → regime\_5d

  * Metrics: AUC, Brier, calibration curves.

* **Explainability**: SHAP (store top contributors per prediction).

---

## **7\) Training & Deployment (SageMaker)**

* **SageMaker Pipeline steps**:

  1. Data prep (read curated.\*, build features\_daily/, labels\_daily/)

  2. Unsupervised fit (GMM) → write regime features

  3. Train/eval (CV, metrics, artifacts)

  4. Register in **Model Registry** with metrics

  5. **Conditional deploy** to:

     * Batch Transform (preferred for cost), or

     * Real-time endpoint (if needed for latency)

* **Artifacts**: versioned in S3; registry tracks model/package versions.

---

## **8\) Serving & APIs**

* **Batch**: Continue Level 2 daily inference to predictions\_daily/.

* **Real-time**: /predict unchanged. (Regime probs may be populated if classifier enabled.)

* **(Optional RAG) Sidecar APIs**:

  * POST /qa → {question} → {answer, citations\[\]}

  * POST /daily-brief → {summary\_md, bullets\[\], citations\[\]}

  * Retrieval filters by recency/source; minimum 2 citations.

---

## **9\) Orchestration & Schedules**

* **Market ETL**: daily 22:05 UTC

* **Macro ETL**: at official release times (e.g., CPI/NFP 13:30 UTC); **forward-fill from next trading day**

* **Features/Labels**: after ETL completion

* **Unsupervised fit**: nightly or weekly (configurable)

* **Train & Eval**: nightly **or on drift**

* **Inference**: nightly batch; *(optional)* intra-day refresh

* **RAG index refresh (optional)**: daily; daily-brief by 07:30 local

---

## **10\) Monitoring, Drift & Promotion**

* **Data quality**: Great Expectations on I/O tables; stop-the-line on contract violations (missing symbols, stale macro).

* **Drift**: PSI/KS on key features & targets; alert thresholds (e.g., PSI \> 0.2).

* **Post-deploy metrics**: rolling RMSE/QLIKE, calibration; CloudWatch dashboards.

* **Promotion rules**: register new model only if:

  * QLIKE (rv) and **Brier** (regime, if enabled) improve beyond threshold (e.g., ≥ 5%) with paired test or min Δ.

* **Shadow/canary**: optional 2–4 weeks; compare live metrics before full traffic shift.

* **Rollback**: pin to previous registry version; toggle env var/alias to revert.

---

## **11\) Security & Compliance**

* **IAM** least-privilege roles; **KMS** encryption for S3/OpenSearch; **Secrets Manager** for API keys.

* **VPC endpoints** for SageMaker & data planes; private subnets for endpoints where possible.

* **Access logs** and audit trails (CloudTrail enabled).

---

## **12\) Non-Functional Requirements**

* **Availability**: ≥ 99% for daily pipeline; predictions by 08:00 local exchange time.

* **Latency**: batch \< 5 min end-to-end; /predict p95 \< 2.5 s.

* **Cost**: ≤ **$X/month** at MVP scale; prefer batch/serverless; right-size instances; auto-stop idle endpoints.

* **Reproducibility**: experiment tracking (SageMaker Experiments or MLflow) \+ artifact hashes.

---

## **13\) Testing**

* **Unit**: feature formulas (macro alignment), leakage checks, regime label construction.

* **Integration**: pipeline run in staging; validate Parquet schemas & partitions.

* **Metrics gates**: training fails if improvement thresholds not met.

* **RAG eval (optional)**: Recall@5 ≥ 0.8 on curated Q/A set; answers include ≥ 2 citations.

---

## **14\) Acceptance Criteria**

* **Macro features** appear in features\_daily/ correctly aligned (no look-ahead).

* **Regime features** (regime\_cluster\_id, prob\_cluster\_\*) populated and used by the regressor.

* **SageMaker Pipeline** registers models with metrics and deploys only on passing gates.

* **Monitoring**: drift and data-quality alarms function; retrain triggers on drift.

* **Performance**: OOS improvement vs Level 2 baseline — e.g., **QLIKE ↓ ≥ 5%** and/or better calibration (documented).

* **(Optional RAG)**: /daily-brief returns summary with ≥ 2 citations; retrieval filters working.

---

## **15\) Runbook (Ops)**

* **Rerun unsupervised step**: trigger SageMaker Processing job; backfill regime features.

* **Pin/rollback model**: set endpoint alias or batch config to prior **Model Registry** version.

* **Drift response**: acknowledge alarm → launch retrain pipeline; review gates & approve promotion.

* **RAG issues**: rebuild index; verify embeddings job & OpenSearch health.

---

## **16\) Resume Bullets (Level 3\)**

* Upgraded the AWS pipeline with **SageMaker Pipelines & Model Registry**, **drift-aware retraining**, and **metrics-gated promotion**, improving **forecast calibration** and **operational reliability**.

* Integrated **macro context** and **unsupervised regime discovery (GMM/HMM)**, delivering statistically significant gains in **QLIKE/RMSE** and better risk-state discrimination.

* Implemented **production monitoring** (Model Monitor, PSI/KS drift, Great Expectations) with automated alerts and rollback workflows.

* *(Optional)* Built a **RAG explainability** sidecar (OpenSearch/pgvector) serving **cited risk briefs** and interactive Q\&A.

