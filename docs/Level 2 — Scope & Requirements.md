# **Level 2 — Scope & Requirements (Markdown)**

## **1\) Summary**

Promote the **Level 1 Docker pipeline** to **AWS** with **scheduled daily inference**, an **internet-facing API**, and **CI/CD** — **no rewrites, same data/APIs**. Use **ECR \+ ECS Fargate/Lambda \+ API Gateway \+ EventBridge**; manage everything with **Terraform**. Keep costs low and reliability high.

---

## **2\) Objectives**

* Run **daily ETL → features → (weekly) train → infer** on AWS, writing Parquet to S3 (same schema as Level 1).

* Expose **/predict** via API Gateway → Lambda (container) or ECS service.

* Stand up **CI/CD** to build/push images and deploy infra.

* Add **observability** (CloudWatch logs/metrics/alarms) and **least-privilege IAM**.

---

## **3\) In Scope**

* **Containerization:** Reuse Level 1 images for ingest, features, train, infer, api.

* **Registry:** ECR repos per service.

* **Compute:** ECS Fargate tasks (or Lambda for light jobs).

* **Orchestration:** EventBridge schedules; optionally Step Functions for stateful chaining.

* **Storage:** S3 Parquet (same layout), optional DynamoDB mirror of predictions\_daily for API reads.

* **API:** API Gateway → Lambda container (FastAPI) or ECS service behind ALB.

* **CI/CD:** GitHub Actions (or CodePipeline) \+ Terraform deploys.

* **Security:** KMS encryption at rest, HTTPS everywhere, IAM least-privilege, secrets in SSM/Secrets Manager.

* **Macro (optional):** Add a small macro\_ingest container (FRED) writing to raw.macro/ and curated.macro\_release\_aligned/ (fields may remain null if you defer to Level 3).

### **Out of Scope (for Level 2\)**

* SageMaker Pipelines / Model Registry / Model Monitor

* Unsupervised regimes (GMM/HMM)

* RAG/LLM components

## **4\) Data Contracts (unchanged)**

s3://\<bucket\>/  
  raw.market/                 (symbol, date, ohlcv)  
  curated.market\_daily/       (symbol, date, ohlc, ret, adj\_close)

  raw.macro/                  (optional at L2)  
  curated.macro\_release\_aligned/ (optional at L2)

  features\_daily/             (date, \<feature columns...\>)  
  labels\_daily/               (date, rv\_5d)  
  predictions\_daily/          (date, rv\_5d\_hat, top\_drivers\[\])

* Append-only; partition by date=YYYY-MM-DD.

* Same /predict request/response as Level 1 (regime fields remain null at this level).

## **5\) Cloud Architecture**

EventBridge (cron)

  └─\> Step Functions (optional) or direct targets

       ├─ ECS Fargate Task: services/ingest

       ├─ ECS Fargate Task: services/features

       ├─ ECS Fargate Task: services/train   (weekly or on Fridays 22:00 UTC)

       └─ ECS Fargate Task: services/infer   (daily 22:30 UTC → S3 predictions)

API Gateway (REST)

  └─\> Lambda (container) running services/api  OR  ECS Fargate service (ALB)

       └─ Reads model from S3 artifacts and returns predictions

       

Data:

  S3 (Parquet, KMS)  \[+ optional DynamoDB mirror of predictions\_daily\]


Observability:

  CloudWatch Logs \+ Metrics \+ Alarms (failed tasks, p95 latency, 5xx rate)


Security:

  IAM least-privilege roles, S3 bucket policies, VPC endpoints (optional), Secrets Manager

## **6\) Schedules & SLAs**

* **Ingest & Features:** Daily at **22:05 UTC** (post close \+ buffer).

* **Train:** Weekly (e.g., **Fri 22:15 UTC**), expanding window; only if metrics gate passes (see §9).

* **Infer:** Daily at **22:30 UTC**; **SLO:** predictions available by **08:00 local exchange time**.

* **API:** Always-on (Lambda on-demand or ECS min=1 task).

---

## **7\) CI/CD & IaC**

**Terraform (infra/)**

* Modules: network/ (optional), s3/, kms/, iam/, ecr/, ecs/, eventbridge/, apigw/, lambda/ (if used), dynamodb/ (optional).

* Outputs: resource ARNs, URLs, bucket names.

**GitHub Actions (or CodePipeline)**

* On push to main:

  1. Run unit/integration tests.

  2. Build containers; tag with commit SHA; push to ECR.

  3. terraform plan \+ terraform apply (with approvals).

  4. Smoke test /predict.

---

## **8\) Service Responsibilities (same code, cloud runners)**

* **ingest**: fetch SPY/VIX/VIX3M/TLT/HYG → raw.market/, curated.market\_daily/

* **features**: compute Level-1 features → features\_daily/

* **train**: weekly training; write artifacts to s3://.../artifacts/model/\<version\>/

* **infer**: daily batch predictions → predictions\_daily/ (+ optional DynamoDB upsert)

* **api**: /predict (FastAPI), uses latest model artifact in S3; returns SHAP top\_drivers

---

## **9\) Quality Gates & Monitoring**

* **Training gate:** Fail deploy of new model if OOS **RMSE/QLIKE** doesn’t beat baseline by ≥ **2–3%**.

* **Canary check:** After new model is active, run a small backfill; alarm if error rate \> threshold.

* **Data validation:** pandera/pydantic schema checks on I/O tables (stop-the-line on contract violations).

* **CloudWatch Alarms:**

  * ECS task failures \> 0 for any step

  * /predict p95 latency \> 2s (Lambda/ECS)

  * 5xx rate \> 1%

  * Missing predictions\_daily partition for latest trading day by **08:00 local**

---

## **10\) Non-Functional Requirements**

* **Availability:** ≥ 99% for daily pipeline (batch) and API.

* **Latency:** /predict p95 \< 2.0 s (Lambda cold starts considered; ECS recommended if needed).

* **Cost target:** ≤ **$X/month** at MVP scale (use Fargate spot for batch, Lambda for API, right-size tasks, lifecycle rules on S3).

* **Security:** KMS for S3, Secrets Manager for API keys, IAM least-privilege, private subnets/VPC endpoints (if feasible).

* **Reproducibility:** Immutable image tags; artifact versioning in S3.

---

## **11\) Testing**

* **Unit tests:** feature formulas, leakage guards, schema validators.

* **Integration tests:** end-to-end “dry run” in a test namespace/bucket.

* **Infra tests:** terraform validate \+ tflint; smoke tests after deploy.

* **Load tests (API):** k6/Locust target p95 \< 2.0 s; concurrency aligned with Lambda/ECS limits.

---

## **12\) Acceptance Criteria**

* **EventBridge** successfully triggers daily **ingest→features→infer**; partitions appear in S3 with yesterday’s date.

* **Weekly train** runs, compares metrics, and only promotes if it passes the gate.

* **API Gateway** /predict returns valid response with SHAP top drivers.

* **CloudWatch** shows logs/metrics; alarms wire to email/Slack (SNS).

* **CI/CD** builds & deploys containers and infra on merge to main.

---

## **13\) Runbook (Ops)**

* **Re-run a failed step:** re-trigger ECS task via Console or aws ecs run-task.

* **Rollback model:** update API/Lambda env var MODEL\_VERSION to previous artifact path; redeploy.

* **Hotfix:** push new image (semantic tag), terraform apply, confirm smoke test, merge to main.

---

## **14\) Cost Controls**

* Use **Fargate Spot** for batch; keep CPU/memory minimal.

* **Lambda** for API (if payloads small) to avoid idle ECS costs.

* S3 lifecycle rules → move old partitions to **IA/Glacier**.

* Stop dev stacks nightly with Terraform workspaces/environments.

---

## **15\) Resume Bullets (Level 2\)**

* Productionized a **Docker-based ML pipeline** on **AWS** using **ECR, ECS Fargate, EventBridge, and API Gateway**, delivering **daily automated forecasts** and a **public /predict API**.

* Implemented **CI/CD** with GitHub Actions \+ **Terraform** for declarative infra; added **CloudWatch** logging/alarms and **IAM least-privilege**.

* Enforced **metrics-based promotion** (RMSE/QLIKE gates) and contract-driven data validation, improving **reliability and observability** while maintaining **low operating costs**.

