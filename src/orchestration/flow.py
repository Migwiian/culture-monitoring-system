#!/usr/bin/env python
"""
Prefect Pipeline: Culture Intelligence Weekly Retraining
--------------------------------------------------------
Orchestrates the end-to-end ML lifecycle for Voluntās.
1. Validation: Ensures data schema integrity.
2. Training: Executes model selection (Linear vs. XGBoost).
3. Promotion: Compares metrics in MLflow and updates production alias.
4. Monitoring: Generates Evidently drift reports.
"""

import subprocess
from pathlib import Path
from prefect import flow, task, get_run_logger
from prefect.schedules import Cron
import mlflow

@task(retries=2, retry_delay_seconds=60)
def validate_data_pipeline(data_path: Path):
    """Task 1: Validate raw data schema before processing."""
    logger = get_run_logger()
    logger.info(f"Validating data schema at {data_path}...")
    
    result = subprocess.run(
        ["python", "src/data/make_dataset.py"],
        capture_output=True,
        text=True,
        check=False
    )
    
    if result.returncode != 0:
        logger.error(f"Data validation failed: {result.stderr}")
        raise RuntimeError("Data pipeline failed")
    
    logger.info("Data validation passed")

@task(retries=2, retry_delay_seconds=60)
def train_model_pipeline():
    """Task 2: Execute training pipeline (Model Selection and Tuning)."""
    logger = get_run_logger()
    logger.info("Training models (Multiple Algorithms)...")
    
    result = subprocess.run(
        ["python", "src/models/train.py"],
        capture_output=True,
        text=True,
        check=False
    )
    
    if result.returncode != 0:
        logger.error(f"Training failed: {result.stderr}")
        raise RuntimeError("Training pipeline failed")
    
    logger.info("Training completed successfully")

@task
def compare_and_promote_model():
    """Task 3: Compare new model with production in MLflow, promote if MAE improves."""
    logger = get_run_logger()
    logger.info("Comparing model performance via MLflow Registry...")
    
    client = mlflow.tracking.MlflowClient()
    # Ensure the experiment name matches your MLflow setup
    experiment = client.get_experiment_by_name("voluntas_culture")
    
    if experiment is None:
        logger.warning("No experiment found, skipping promotion logic")
        return
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=2
    )
    
    if len(runs) < 2:
        logger.info("Initial run detected. Promoting to production alias.")
        client.set_registered_model_alias("culture_model", "production", runs[0].info.run_id)
        return
    
    # Selection based on MAE (Mean Absolute Error)
    latest_mae = runs[0].data.metrics.get("mae", 999)
    previous_mae = runs[1].data.metrics.get("mae", 999)
    
    if latest_mae < previous_mae:
        logger.info(f"Performance improved: {latest_mae:.3f} < {previous_mae:.3f}. Promoting new version.")
        client.set_registered_model_alias("culture_model", "production", runs[0].info.run_id)
    else:
        logger.info(f"Retention: Previous model remains production baseline ({previous_mae:.3f}).")

@task
def generate_drift_report():
    """Task 4: Execute Evidently drift analysis."""
    logger = get_run_logger()
    logger.info("Generating data drift report...")
    
    result = subprocess.run(
        ["python", "src/monitoring/drift_report.py"],
        capture_output=True,
        text=True,
        check=False
    )
    
    if result.returncode == 0:
        logger.info("Drift report generated in /reports")
    else:
        logger.warning("Drift report execution encountered errors, continuing flow.")

@flow(name="culture-intelligence-weekly", 
      description="Weekly retraining pipeline for Voluntās culture monitoring")
def weekly_training_flow():
    """Main flow orchestrator for the Voluntās ML pipeline."""
    logger = get_run_logger()
    logger.info("Starting weekly culture intelligence pipeline")
    
    # Resolve directory structure
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / "data" / "processed"
    
    # Sequential Pipeline Execution
    validate_data_pipeline(data_path)
    train_model_pipeline()
    compare_and_promote_model()
    generate_drift_report()
    
    logger.info("Weekly pipeline execution finalized.")

if __name__ == "__main__":
    """
    Deployment Configuration
    ------------------------
    Utilizes .serve() to create a long-running process that listens 
    for the specified Cron schedule. 
    Schedule: Every Monday at 09:00 AM.
    """
    weekly_training_flow.serve(
        name="weekly-retraining-deployment",
        schedule=Cron("0 9 * * 1")
    )