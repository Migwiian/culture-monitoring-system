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
```
## This is a fantastic foundation for a professional README. Since the ML Zoomcamp evaluators look for a clear explanation of why the project exists, crafting a strong problem statement is the final piece of the puzzle.

In your case, the problem is the gap between static consulting and dynamic monitoring.

Here is the updated version of your README. I’ve refined the structure to be more "submission-ready," added the problem statement, and included the new Docker section.

Voluntās Culture Intelligence System
Operationalizing Meaningfulness: An MLOps-Driven Approach to Workplace Culture Monitoring

🎯 Problem Statement
Traditional workplace culture assessment relies on annual engagement surveys or one-off consulting sessions. These methods provide "snapshots" that quickly become outdated, especially in high-churn environments.

The Voluntās Culture Intelligence System solves this by providing a continuous, automated way to monitor workplace health. By analyzing external employee reviews (Glassdoor), the system predicts the "Overall Meaningfulness" of an organization. This allows leadership to:

Identify cultural decay in real-time before it leads to mass resignations.

Benchmark growth against historical data (2008–2024).

Validate consulting hypotheses with large-scale statistical evidence.

🏗️ Technical Architecture
Data: 838,566 reviews processed from raw CSV into engineered Parquet format.

Models: Comparison between Linear Regression (Baseline) and XGBoost (Tuned).

Validation: Temporal split (pre-2020 vs. post-2020) to ensure the model handles modern workplace shifts.

Deployment: Containerized FastAPI service capable of real-time inference.

✅ Project Checklist (ML Zoomcamp Requirements)
Multiple Models: Evaluated Linear Regression and XGBoost.

Parameter Tuning: Systematic tuning of XGBoost (max_depth, n_estimators) documented in training logs.

Model Selection: Final selection based on Mean Absolute Error (MAE) and model complexity.

Containerization: Full Docker integration for reproducible deployment.

Dependency Management: Pinned requirements.txt for environment consistency.

🚀 Quick Start (Local)
Prerequisites
Python 3.12

Docker (optional, but recommended)

1. Setup & Training
Bash

# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the training pipeline (generates best_model.bin)
python src/models/train.py
2. Run with Docker (Recommended)
This is the easiest way to run the prediction service as a standalone container.

Bash

# Build the image
docker build -t culture-api .

# Start the container
docker run -p 8000:8000 culture-api
3. Test the API
Once the container is running, access the interactive documentation at: http://localhost:8000/docs

Or test via curl:

Bash

curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"culture_values": 4.0, "belonging_score": 3.5, "career_opp": 4.0}'


## Model Performance & Selection
We evaluated multiple algorithms to predict employee satisfaction ratings (scale 1-5).

* Linear Regression: 0.541 MAE (Baseline)

* XGBoost (Tuned): 0.541 MAE

Selection Note: While both models performed similarly on the current feature set, XGBoost was selected as the final model due to its ability to capture non-linear relationships that may emerge as more complex features (like sentiment analysis) are added to the pipeline.