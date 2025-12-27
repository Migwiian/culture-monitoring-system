#!/usr/bin/env python
"""
Module: make_dataset.py
Description: Production pipeline for Voluntās Culture Intelligence System.
Engineers meaningfulness features from 850K Glassdoor reviews for regression modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_raw_data(input_path: Path) -> pd.DataFrame:
    """Performs schema validation on the raw input file."""
    logger.info(f"Validating {input_path}")
    
    if not input_path.exists():
        logger.error(f"❌ File not found: {input_path}")
        raise FileNotFoundError(f"Missing raw data file at {input_path}")

    # Load sample to check structure
    df_sample = pd.read_csv(input_path, nrows=100)
    logger.info(f"Available columns: {df_sample.columns.tolist()}")

    # Required columns for Voluntās pillars
    required_cols = [
        'culture_values', 'work_life_balance', 'senior_mgmt', 
        'career_opp', 'diversity_inclusion', 'overall_rating'
    ]
    
    missing = [col for col in required_cols if col not in df_sample.columns]
    if missing:
        raise ValueError(f"❌ Input data is missing required pillars: {missing}")
    
    logger.info("Schema validation successful.")
    return df_sample

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers Voluntās Culture Pillars and features for regression."""
    logger.info("Engineering Voluntās Culture Intelligence...")

    # 1. CRITICAL: Drop rows without target (REGRESSION REQUIRES THIS)
    initial_rows = len(df)
    df = df.dropna(subset=['overall_rating'])
    logger.info(f"Dropped {initial_rows - len(df):,} rows with missing ratings. Remaining: {len(df):,}")

    # 2. TEMPORAL: Convert date for trend monitoring
    if 'date_review' in df.columns:
        df['date_review'] = pd.to_datetime(df['date_review'], errors='coerce')
        df['days_since_review'] = (pd.Timestamp.now() - df['date_review']).dt.days
        logger.info("Added temporal features")

    # 3. OPTIMIZATION: Convert strings to categories (Saves RAM on 850K rows)
    for col in ['firm', 'job_title', 'location']:
        if col in df.columns:
            df[col] = df[col].astype('category')
            logger.info(f"Optimized '{col}' as category")

    # 4. MEANINGFULNESS INDEX: Weighted calculation (Voluntās methodology)
    belonging_cols = ['work_life_balance', 'senior_mgmt', 'diversity_inclusion']
    if all(col in df.columns for col in belonging_cols):
        df['belonging_score'] = df[belonging_cols].mean(axis=1)
        
        df['voluntas_index'] = (
            df['culture_values'] * 0.4 + 
            df['belonging_score'] * 0.3 +
            df['career_opp'] * 0.3
        )
        logger.info("Calculated Voluntās Meaningfulness Index")
    else:
        logger.warning("Missing belonging columns - index skipped")

    # 5. TEXT SIGNALS: Safe engagement metric (if text exists)
    text_cols = ['pros', 'cons']
    if all(col in df.columns for col in text_cols):
        df['pros_length'] = df['pros'].fillna('').str.len()
        df['cons_length'] = df['cons'].fillna('').str.len()
        df['engagement_signal'] = df['pros_length'] - df['cons_length']  # Net sentiment
        logger.info("Added text engagement signals")
    else:
        logger.info("No pros/cons columns found - skipping text features")

    # 6. FINAL LOGGING
    new_cols = [col for col in df.columns if '_score' in col or '_signal' in col or '_index' in col]
    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    logger.info(f"New engineered columns: {new_cols}")

    return df

if __name__ == "__main__":
    # Robust path resolution (works from any directory)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "glassdoor_reviews.csv"
    PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "culture_intelligence_v1.parquet"

    try:
        # Ensure output directory exists
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

        # 1. Validate schema
        validate_raw_data(RAW_DATA_PATH)

        # 2. Load full dataset
        logger.info("Loading full 850K dataset...")
        df_raw = pd.read_csv(RAW_DATA_PATH)

        # 3. Clean & engineer
        df_processed = clean_data(df_raw)

        # 4. Save to parquet (fast, compressed, production-ready)
        df_processed.to_parquet(PROCESSED_DATA_PATH, index=False, compression="snappy")
        logger.info(f"Clean data saved to: {PROCESSED_DATA_PATH}")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise