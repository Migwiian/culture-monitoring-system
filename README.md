# Voluntās Culture Intelligence System

**Operationalizing Meaningfulness: An MLOps-Driven Approach to Workplace Culture Monitoring**

---

## Project Inspiration

This system draws from Voluntās' philosophy of measuring workplace meaningfulness through four core pillars—Purpose, Belonging, Growth, and Leadership—and operationalizes it as a production ML pipeline. Rather than replacing Voluntās' methodology, we scale it: transforming annual consulting insights into continuous, data-driven culture monitoring.

---

## Current State

**Built:**
- **Data Pipeline:** Processes 838K Glassdoor reviews (2008-2024) into engineered features
- **Feature Engineering:** Voluntās Meaningfulness Index (weighted composite of culture pillars)
- **Model Comparison:** LinearRegression + 2 XGBoost variants with temporal validation
- **Experiment Tracking:** MLflow logs all runs, parameters, and artifacts
- **Leakage Protection:** Train-only imputation, temporal splits, feature flags
- **Orchestration:** Prefect flow structure (requires Prefect 3.x compatibility fixes)

**Pending:**
- Service deployment (FastAPI)
- Drift monitoring (Evidently)
- CI/CD pipeline (GitHub Actions)
- Automated weekly retraining schedule

---

## Architecture

```mermaid
flowchart TD
    A[Raw Reviews] --&gt; B[Feature Engineering]
    B --&gt; C[Temporal Split]
    C --&gt; D[Train Models]
    D --&gt; E[MLflow Tracking]
    E --&gt; F[Prefect Orchestration]
    F --&gt; G[FastAPI Service]
    G --&gt; H[Evidently Monitoring]
```

## Quick Start
``` bash
# Setup environment
make setup

# Process data (requires data/raw/glassdoor_reviews.csv)
make data

# Train models (logs to MLflow)
make train

# View results
# open http://localhost:5000
```

## Technical Implementation
* Data: 838,566 reviews, 28 engineered features, 0% missing values
* Models: LinearRegression + XGBoost (2 hyperparameter sets)
* Validation: Temporal split (2020-11-19 cutoff), MAE=0.562
* Orchestration: Prefect 3.x compatible flow structure
* Tracking: MLflow at http://localhost:5000

## Repository Structure
├── data/
│   ├── raw/          # 850K Glassdoor CSV (not committed)
│   └── processed/    # Parquet files (generated)
├── src/
│   ├── data/         # make_dataset.py
│   ├── models/       # train.py (multiple models)
│   ├── orchestration/# flow.py (Prefect)
│   └── monitoring/   # drift detection (TODO)
├── deployment/       # FastAPI service (TODO)
├── models/           # Serialized models
└── tests/            # Unit tests (TODO)