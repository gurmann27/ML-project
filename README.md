---
title: ML Project
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlet
sdk_version: 1.55.0
app_file: streamlit_app.py
pinned: false
---
# 🧠 Customer Monitoring System — Backend

A production-grade ML backend for monitoring customer health, predicting churn, detecting anomalies, and analyzing sentiment — powered by **XGBoost** (classification) and **Transformer NLP**, built with **FastAPI**.

---

## ML Problem Types

| Problem | Algorithm | Type |
|---|---|---|
| Churn Prediction | XGBoost (binary classification) | Classification |
| Customer Segmentation | XGBoost (multiclass) | Classification |
| Anomaly Detection | Isolation Forest | Unsupervised |
| Sentiment Analysis | RoBERTa (transformer) | NLP |
| Feature Engineering | SHAP + TF-IDF signals | ML + NLP hybrid |

---

## Project Structure

```
customer-monitoring-backend/
├── app/
│   ├── main.py                          # FastAPI app entry point
│   ├── core/
│   │   ├── config.py                    # Settings (env vars)
│   │   ├── model_registry.py            # Loads & manages all models
│   ├── models/
│   │   └── xgboost_trainer.py           # XGBoost churn, segment, anomaly trainers
│   ├── schemas/
│   │   └── schemas.py                   # All Pydantic request/response schemas
│   ├── services/
│   │   ├── monitoring_service.py        # Business logic orchestration
│   │   ├── nlp_service.py               # Transformer NLP pipeline
│   │   └── drift_service.py             # KS test + PSI data drift detection
│   └── api/v1/
│       ├── router.py                    # Route aggregator
│       └── endpoints/
│           ├── health.py                # GET /health
│           ├── churn.py                 # POST /churn/predict
│           ├── segmentation.py          # POST /segment/predict
│           ├── anomaly.py               # POST /anomaly/detect
│           ├── sentiment.py             # POST /sentiment/analyze
│           ├── reports.py               # POST /report/full
│           ├── training.py              # POST /train/churn|segment|anomaly
│           └── monitoring.py            # GET /monitoring/dashboard
├── mlops/
│   ├── docker/
│   │   ├── Dockerfile                   # Multi-stage Docker build
│   │   ├── docker-compose.yml           # API + Postgres + MLflow + pgAdmin
│   │   └── kubernetes.yml               # K8s Deployment + HPA + Ingress
│   └── ci_cd/
│       └── github-actions.yml           # Full CI/CD pipeline
├── tests/
│   └── test_api.py                      # 20+ endpoint tests (all mocked)
├── scripts/
│   └── generate_sample_data.py          # Synthetic dataset generator
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd customer-monitoring-backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env

# 3. Generate sample training data
python scripts/generate_sample_data.py

# 4. Run the server
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## Train the Models

```bash
# 1. Upload CSV and train churn XGBoost model
curl -X POST http://localhost:8000/api/v1/train/churn \
  -F "file=@data/sample_customers.csv"

# 2. Train segmentation model
curl -X POST http://localhost:8000/api/v1/train/segment \
  -F "file=@data/sample_customers.csv"

# 3. Train anomaly detector
curl -X POST http://localhost:8000/api/v1/train/anomaly \
  -F "file=@data/sample_customers.csv"
```

---

## API Reference

### Churn Prediction (XGBoost Classification)

```bash
POST /api/v1/churn/predict
POST /api/v1/churn/predict/batch    # up to 1000 customers
```

**Response includes:** churn probability, risk level (low/medium/high/critical), SHAP-based top risk factors, recommended retention actions.

### Segmentation (XGBoost Multiclass)

```bash
POST /api/v1/segment/predict
POST /api/v1/segment/predict/batch
```

**Segments:** Champion, Loyal, At-Risk, Hibernating, New, Potential — with RFM scores and upsell opportunities.

### Anomaly Detection

```bash
POST /api/v1/anomaly/detect
```

**Detects:** payment anomalies, support abuse patterns, long inactivity, behavioral outliers.

### Sentiment Analysis (NLP)

```bash
POST /api/v1/sentiment/analyze         # single text
POST /api/v1/sentiment/analyze/batch   # batch (up to 500)
```

### Full Report (All Models)

```bash
POST /api/v1/report/full
```

Returns unified customer health report: churn + segment + anomaly + sentiment + composite health score (0–100).

### MLOps

```bash
GET  /api/v1/monitoring/dashboard      # model status
POST /api/v1/monitoring/drift          # data drift check (upload CSV)
POST /api/v1/monitoring/drift/set-reference
```

---

## MLOps Stack

| Tool | Purpose |
|---|---|
| **Git + GitHub** | Version control |
| **GitHub Actions** | CI/CD (lint → test → build → deploy) |
| **Docker** | Containerization (multi-stage build) |
| **Kubernetes** | Orchestration (Deployment + HPA + Ingress) |
| **MLflow** | Experiment tracking, model registry |
| **SHAP** | Model explainability |
| **KS Test + PSI** | Data drift detection |
| **Prometheus** | Metrics instrumentation |

---

## Deployment Platforms

The CI/CD pipeline supports deploying to **AWS ECS**, **GCP Cloud Run**, or **Render** — set `DEPLOY_TARGET` in GitHub environment variables.

### Docker Compose (local)
```bash
cd mlops/docker
docker-compose up --build
```

### Kubernetes
```bash
kubectl apply -f mlops/docker/kubernetes.yml
```

---

## Testing

```bash
pytest tests/ -v --asyncio-mode=auto
```

All tests use mocked ML models — no GPU or HuggingFace downloads required for testing.

---

## Form Answers

| Question | Answer |
|---|---|
| **Q8 — ML/DL Problem Type** | Classification (XGBoost binary churn + multiclass segmentation) + NLP (sentiment) |
| **Q12 — MLOps** | Git/GitHub · CI/CD via GitHub Actions · Containerized with Docker + Kubernetes |
| **Q13 — Deployment Platform** | AWS ECS / GCP Cloud Run / Render / Local (Docker Compose) |
