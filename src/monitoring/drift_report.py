# src/monitoring/drift_report.py
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_drift_report():
    """Generate Evidently drift report (placeholder for now)."""
    logger.info("Generating drift report placeholder")
    # TODO: Load train vs. new data, create Evidently report
    Path("reports/drift_report.html").touch()
    logger.info("Placeholder report created")

if __name__ == "__main__":
    generate_drift_report()