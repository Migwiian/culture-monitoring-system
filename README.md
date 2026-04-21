# Voluntās Culture Intelligence System

**Operationalizing Meaningfulness: An MLOps-Driven Approach to Workplace Culture Monitoring**

## Problem Statement: The Gap Between Strategy and Soul
Most organizations treat culture as a soft metric, measured through infrequent surveys that offer little more than a post-mortem of employee dissatisfaction. This creates a dangerous "Culture Lag"—where leadership is blind to the erosion of meaningfulness until it manifests as turnover or decreased productivity.

Inspired by the Voluntās philosophy, this system addresses the fundamental right to a meaningful workplace. By operationalizing the Meaningful Work Quotient (MWQ) through a production-grade ML pipeline, we bridge the gap between existential philosophy and data-driven action. We transform raw, unstructured employee sentiment into a continuous stream of insights, enabling leaders to proactively nurture Purpose, Belonging, Growth, and Leadership.

## Technical Architecture
* Data: 838,566 reviews processed from raw CSV into engineered Parquet format (latest + date-stamped version).
* Models: Comparison between Linear Regression (Baseline) and XGBoost (Tuned).
* Validation: Temporal split by review date (80/20) to reduce leakage and reflect real-world shifts.
* Deployment: Containerized FastAPI service capable of real-time inference.
* Orchestration: The entire ML pipeline is orchestrated using Kestra.

### Pipeline Overview
```
+---------------------+      +-------------------------+      +--------------------+      +---------------------------+
| Git Clone           |----->| Install Dependencies    |----->| Process Data       |----->| Train Model               |
| (Project Repo)      |      | (requirements.txt)      |      | (make_dataset.py)  |      | (train.py)                |
+---------------------+      +-------------------------+      +--------------------+      +---------------------------+
                                                                                                  |
                                                                                                  |
                                                                                                  v
+-----------------------------+
| Generate Drift Report       |
| (drift_report.py)           |
+-----------------------------+
```

## Quick Start (local)
Prerequisites
    * Python 3.12
    * Docker (optional, but recommended)
    * Kestra (for orchestration)

### 1. Data Acquisition
The `glassdoor_reviews.csv` dataset is required and should be placed in the `src/data/` directory. Processed outputs are written to `src/data/processed/` as both `culture_intelligence_v1.parquet` (latest) and a date-stamped version.

To acquire the dataset, you can use the Kaggle API:
1.  **Install Kaggle API:**
    ```bash
    pip install kaggle
    ```
2.  **Set up Kaggle API credentials:**
    *   Go to your Kaggle account page and create a new API token. This will download a `kaggle.json` file.
    *   Place this file in `~/.kaggle/` and set the permissions:
        ```bash
        mkdir -p ~/.kaggle
        mv kaggle.json ~/.kaggle/
        chmod 600 ~/.kaggle/kaggle.json
        ```
3.  **Download and place the dataset:**
    ```bash
    kaggle datasets download -d sagar-0817/glassdoor_reviews -p src/data/ --unzip
    ```

### 2. Local Setup and Training
``` bash
# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the training pipeline (generates best_model.bin)
python src/models/train.py
```

### 3. Orchestration with Kestra
This project uses Kestra to orchestrate the entire ML pipeline, from data processing to model training and monitoring.

**To run the pipeline with Kestra:**
1.  **Install Kestra:** Follow the official Kestra installation guide: [https://kestra.io/docs/getting-started/installation](https://kestra.io/docs/getting-started/installation)
2.  **Start Kestra:** Run the Kestra server.
3.  **Create the Flow:** In the Kestra UI, navigate to "Flows" and create a new flow, pasting the contents of `src/orchestration/kestra.yml`.
4.  **Run the Flow:** You can trigger the flow manually from the Kestra UI.

### 4. Run with Docker
``` bash
# Build the image
docker build -t culture-api .

# Start the container
docker run -p 8000:8000 culture-api
```

### 5. Test the API
Once the container is running, access the interactive documentation at: http://localhost:8000/docs

Or test via curl:

``` Bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"culture_values": 4.0, "belonging_score": 3.5, "career_opp": 4.0}'
```

### 6. Cloud Deployment with Render
This project can be deployed to Render using the provided `render.yaml` configuration.

1.  **Fork the repository:** Fork this GitHub repository to your own account.
2.  **Create a new Web Service on Render:**
    *   Go to your Render Dashboard and click "New Web Service".
    *   Select your forked GitHub repository.
    *   Render will automatically detect the `render.yaml` file in the `deployment/` directory. Confirm the settings.
    *   Click "Create Web Service".
3.  **Access the deployed API:** Once deployed, Render will provide a live URL for your service. You can access the interactive documentation at `YOUR_RENDER_URL/docs`.

## Model Performance & Selection
We evaluated multiple algorithms to predict employee satisfaction ratings (scale 1-5).

* Linear Regression: 0.541 MAE (Baseline)
* XGBoost (Tuned): 0.541 MAE

Selection Note: While both models performed similarly on the current feature set, XGBoost was selected as the final model due to its ability to capture non-linear relationships that may emerge as more complex features (like sentiment analysis) are added to the pipeline.

## MWQ Proxy (Single Strategy)
This project uses a single meaningfulness metric aligned to the Voluntās MWQ framework using dataset proxies:
* Purpose = `culture_values`
* Leadership = `senior_mgmt`
* Belonging = mean(`work_life_balance`, `diversity_inclusion`)
* Growth = `career_opp`

MWQ Proxy = mean(Purpose, Leadership, Belonging, Growth)

## Monitoring
The `src/monitoring/drift_report.py` script is a placeholder for future implementation of data and model drift monitoring. This is a crucial component of a production MLOps system to ensure the model's performance doesn't degrade over time.
