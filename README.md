# Voluntās Culture Intelligence System

**Operationalizing Meaningfulness: An MLOps-Driven Approach to Workplace Culture Monitoring**

## Problem Statement: The Gap Between Strategy and Soul
Most organizations treat culture as a "soft" metric, measured through infrequent surveys that offer little more than a post-mortem of employee dissatisfaction. This creates a dangerous "Culture Lag"—where leadership is blind to the erosion of meaningfulness until it manifests as turnover or decreased productivity.

Inspired by the Voluntās philosophy, this system addresses the fundamental right to a meaningful workplace. By operationalizing the Meaningful Work Quotient (MWQ) through a production-grade ML pipeline, we bridge the gap between existential philosophy and data-driven action. We transform raw, unstructured employee sentiment into a continuous stream of insights, enabling leaders to proactively nurture Purpose, Belonging, Growth, and Leadership.

## Technical Architecture
* Data: 838,566 reviews processed from raw CSV into engineered Parquet format.

* Models: Comparison between Linear Regression (Baseline) and XGBoost (Tuned).

* Validation: Temporal split (pre-2020 vs. post-2020) to ensure the model handles modern workplace shifts.

* Deployment: Containerized FastAPI service capable of real-time inference.

## Quick Start(local)
Prerequisites
    * Python 3.12
    * Docker (optional, but recommended)
1. Setup and training
``` bash
    # Setup environment
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    #Run the training pipeline (generates best_model.bin)
    python src/models/train.py
``` 
2. Run with Docker
``` bash
# Build the image
docker build -t culture-api .

# Start the container
docker run -p 8000:8000 culture-api
```
3. Test the API
Once the container is running, access the interactive documentation at: http://localhost:8000/docs

Or test via curl:

``` Bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"culture_values": 4.0, "belonging_score": 3.5, "career_opp": 4.0}'
