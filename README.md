# Voluntās Culture Intelligence System

**Operationalizing Global Meaningfulness Through MLOps**

---

## 🎯 Problem

Voluntās advises Fortune 500 companies on workplace culture, but their manual survey methodology is:
- **Slow**: Annual surveys = 6-month lag time
- **Expensive**: $200-500/hour consulting rates limit scale
- **Reactive**: Advisors can't prevent talent loss, only report it

## 💡 Solution

A **production ML pipeline** that continuously monitors employee sentiment across 850K+ Glassdoor reviews, predicts culture degradation, and alerts consultants in real-time.

---

## 🏗️ Architecture

- **Data**: 850K Glassdoor reviews (global, 2008-2024)
- **Orchestration**: Prefect for weekly retraining
- **Experiment Tracking**: MLflow for model comparison
- **Deployment**: FastAPI microservice for predictions
- **Monitoring**: Evidently + Grafana for drift detection
- **CI/CD**: GitHub Actions for automated deployment

---

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone &lt;repo&gt;
cd voluntas-culture-intelligence
make setup

# 2. Place your 850K CSV in data/raw/
# File: glassdoor_reviews.csv

# 3. Validate & clean data
make data

# 4. Train model
make train

# 5. Run API
make api
# → http://localhost:9696/predict/Tesla

# 6. Deploy full stack
make deploy