#!/usr/bin/env python
"""
Module: make_dataset.py
Description: Production data pipeline for Voluntās Culture Intelligence System.
Engineers meaningfulness features from 850K Glassdoor reviews for regression modeling 
with robust missing value handling.
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
    """
    Validates that the raw CSV contains all required columns for Voluntās pillars.
    
    Args:
        input_path: Path to raw Glassdoor reviews CSV
        
    Returns:
        DataFrame sample for schema inspection
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If required columns are missing
    """
    logger.info(f"Validating {input_path}")
    
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        raise FileNotFoundError(f"Missing raw data file at {input_path}")

    # Load sample to check structure without reading full dataset
    df_sample = pd.read_csv(input_path, nrows=100)
    logger.info(f"Available columns: {df_sample.columns.tolist()}")

    # Required columns for Voluntās pillars
    required_columns = [
        'overall_rating',      # Regression target
        'culture_values',      # Purpose proxy
        'work_life_balance',   # Belonging component
        'senior_mgmt',         # Belonging component  
        'diversity_inclusion', # Belonging component
        'career_opp'           # Growth proxy
    ]
    
    missing_cols = [col for col in required_columns if col not in df_sample.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info("Schema validation successful.")
    return df_sample


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers Voluntās Culture Intelligence features with robust missing value handling.
    
    Strategy:
    1. Preserve all rows with target variable (regression requirement)
    2. Handle partial missingness in belonging pillars using skipna=True
    3. Impute complete missingness with global median (conservative fallback)
    4. Flag imputed rows for production monitoring
    
    Args:
        df: Raw DataFrame with Glassdoor review columns
        
    Returns:
        DataFrame with engineered features and quality flags
    """
    logger.info("Engineering Voluntās Culture Intelligence...")

    # ----------------------------------------------------------------------
    # TARGET: Drop only rows without the regression target
    # ----------------------------------------------------------------------
    initial_row_count = len(df)
    df = df.dropna(subset=['overall_rating'])
    logger.info(f"Target preservation: {initial_row_count:,} -> {len(df):,} rows")
    
    # ----------------------------------------------------------------------
    # TEMPORAL: Add time-based features for drift detection
    # ----------------------------------------------------------------------
    if 'date_review' in df.columns:
        df['date_review'] = pd.to_datetime(df['date_review'], errors='coerce')
        df['days_since_review'] = (pd.Timestamp.now() - df['date_review']).dt.days
        logger.info("Added temporal features")
    
    # ----------------------------------------------------------------------
    # CATEGORICAL: Optimize memory for high-cardinality string columns
    # ----------------------------------------------------------------------
    categorical_cols = ['firm', 'job_title', 'location']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            logger.info(f"Optimized '{col}' as category")
    
    # ----------------------------------------------------------------------
    # BELONGING SCORE: Handle missing pillar data intelligently
    # ----------------------------------------------------------------------
    belonging_pillars = ['work_life_balance', 'senior_mgmt', 'diversity_inclusion']
    
    # Phase 1: Use available data (partial missingness OK)
    df['belonging_score'] = df[belonging_pillars].mean(axis=1, skipna=True)
    
    # Phase 2: Impute rows where ALL pillars are missing
    completely_missing_mask = df[belonging_pillars].isna().all(axis=1)
    
    if completely_missing_mask.any():
        missing_rate = completely_missing_mask.mean()
        logger.warning(f"{missing_rate:.1%} rows have ZERO belonging data - using fallback")
        
        # Conservative imputation: Global median from rows with data
        global_median = df.loc[~completely_missing_mask, 'belonging_score'].median()
        df.loc[completely_missing_mask, 'belonging_score'] = global_median
        
        # Flag for production monitoring
        df['belonging_imputed'] = completely_missing_mask
        logger.info(f"Imputed with global median: {global_median:.2f}")
    else:
        df['belonging_imputed'] = False
    
    # Flag partial missingness (firms hiding DEI data)
    df['belonging_incomplete'] = df[belonging_pillars].isna().any(axis=1)


    # ----------------------------------------------------------------------
    # PURPOSE & GROWTH: Handle single-column pillars with flags
    # ----------------------------------------------------------------------
    for pillar in ['culture_values', 'career_opp']:
        if df[pillar].isna().any():
            missing_rate = df[pillar].isna().mean()
            logger.warning(f"{pillar}: {missing_rate:.1%} missing")
        
            # Flag before imputation (MLOps best practice)
            df[f'{pillar}_imputed'] = df[pillar].isna()
        
            # Median imputation
            pillar_median = df[pillar].median()
            df[pillar] = df[pillar].fillna(pillar_median)
            logger.info(f"Imputed {pillar} with median: {pillar_median:.2f}")
        else:
            df[f'{pillar}_imputed'] = False
    # ----------------------------------------------------------------------
    # VOLUNTĀS INDEX: Weighted composite metric (Purpose 40%, Belonging 30%, Growth 30%)
    # ----------------------------------------------------------------------
    df['voluntas_index'] = (
        df['culture_values'] * 0.4 + 
        df['belonging_score'] * 0.3 +
        df['career_opp'] * 0.3
    )
    logger.info("Calculated Voluntās Meaningfulness Index")
    
    # ----------------------------------------------------------------------
    # TEXT SIGNALS: Net sentiment proxy from review word counts
    # ----------------------------------------------------------------------
    text_columns = ['pros', 'cons']
    if all(col in df.columns for col in text_columns):
        df['pros_length'] = df['pros'].fillna('').str.len()
        df['cons_length'] = df['cons'].fillna('').str.len()
        df['engagement_signal'] = df['pros_length'] - df['cons_length']
        logger.info("Added text engagement signals")
    else:
        logger.info("No pros/cons columns found - skipping text features")
    
    # ----------------------------------------------------------------------
    # FINAL VALIDATION: Ensure engineered features are present
    # ----------------------------------------------------------------------
    engineered_features = [
        'voluntas_index', 'belonging_score', 'engagement_signal',
        'belonging_incomplete', 'belonging_imputed'
    ]
    
    present_features = [f for f in engineered_features if f in df.columns]
    
    logger.info("Feature Engineering Summary:")
    logger.info(f"  Final shape: {df.shape}")
    logger.info(f"  Engineered features: {present_features}")
    
    return df


def main():
    """Main execution pipeline with error handling"""
    # Use robust path resolution (works from any directory)
    project_root = Path(__file__).resolve().parent.parent.parent
    
    input_path = project_root / "data" / "raw" / "glassdoor_reviews.csv"
    output_path = project_root / "data" / "processed" / "culture_intelligence_v1.parquet"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Validate schema before expensive processing
        validate_raw_data(input_path)
        
        # 2. Load full dataset
        logger.info(f"Loading dataset: {input_path.name}")
        df_raw = pd.read_csv(input_path)
        
        # 3. Engineer features
        df_processed = clean_data(df_raw)
        
        # 4. Save to Parquet (fast, compressed, preserves dtypes)
        logger.info(f"Saving to: {output_path.name}")
        df_processed.to_parquet(
            output_path,
            index=False,
            compression="snappy"
        )
        
        logger.info("Pipeline completed successfully")
        logger.info(f"Output: {len(df_processed):,} rows, {df_processed.shape[1]} columns")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()