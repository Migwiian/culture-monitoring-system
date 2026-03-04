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
import sys
import json
from datetime import date
from pandera import Column, DataFrameSchema, Check
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_FILENAME,
    PROCESSED_DATA_FILENAME,
)

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
        "overall_rating",      # Regression target
        "culture_values",      # Purpose proxy
        "work_life_balance",   # Belonging component
        "senior_mgmt",         # Belonging component
        "diversity_inclusion", # Belonging component
        "career_opp",          # Growth proxy
    ]
    
    missing_cols = [col for col in required_columns if col not in df_sample.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Lightweight schema validation (coerce numerics + range checks)
    schema = DataFrameSchema(
        {
            "overall_rating": Column(
                float,
                coerce=True,
                nullable=False,
                checks=Check.in_range(1.0, 5.0),
            ),
            "culture_values": Column(float, coerce=True, nullable=True, checks=Check.in_range(1.0, 5.0)),
            "work_life_balance": Column(float, coerce=True, nullable=True, checks=Check.in_range(1.0, 5.0)),
            "senior_mgmt": Column(float, coerce=True, nullable=True, checks=Check.in_range(1.0, 5.0)),
            "diversity_inclusion": Column(float, coerce=True, nullable=True, checks=Check.in_range(1.0, 5.0)),
            "career_opp": Column(float, coerce=True, nullable=True, checks=Check.in_range(1.0, 5.0)),
        },
        strict=False,
    )
    schema.validate(df_sample, lazy=True)

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
    # Drop rows without overall_rating (regression target)
    df = df.dropna(subset=['overall_rating'])
    logger.info(f"Target preservation: {initial_row_count:,} -> {len(df):,} rows")

    # ----------------------------------------------------------------------
    # TEMPORAL: Add time-based features for drift detection
    # ----------------------------------------------------------------------
    if 'date_review' in df.columns:
        try:
            # Convert date_review to datetime format
            df['date_review'] = pd.to_datetime(df['date_review'], errors='coerce')
        except ValueError as e:
            logger.error(f"Failed to parse date_review column: {str(e)}")
        else:
            # Calculate days since review for drift detection
            df['days_since_review'] = (pd.Timestamp.now() - df['date_review']).dt.days
            logger.info("Added temporal features")

    # ----------------------------------------------------------------------
    # CATEGORICAL: Optimize memory for high-cardinality string columns
    # ----------------------------------------------------------------------
    categorical_cols = ['firm', 'job_title', 'location']
    for col in categorical_cols:
        if col in df.columns:
            try:
                # Convert categorical columns to category dtype
                df[col] = df[col].astype('category')
            except TypeError as e:
                logger.error(f"Failed to convert {col} to category: {str(e)}")
            else:
                logger.info(f"Optimized '{col}' as category")

    # ----------------------------------------------------------------------
    # BELONGING SCORE: Handle missing pillar data intelligently
    # ----------------------------------------------------------------------
    belonging_pillars = ['work_life_balance', 'senior_mgmt', 'diversity_inclusion']
    
    # Phase 1: Use available data (partial missingness OK)
    # Calculate mean of available belonging pillars
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
        
            # Flag before imputation (MLOPS best practice)
            df[f'{pillar}_imputed'] = df[pillar].isna()
        
            # Median imputation
            pillar_median = df[pillar].median()
            df[pillar] = df[pillar].fillna(pillar_median)
            logger.info(f"Imputed {pillar} with median: {pillar_median:.2f}")
        else:
            df[f'{pillar}_imputed'] = False
    # ----------------------------------------------------------------------
    # MWQ PROXY: Four Voluntas pillars (Purpose, Leadership, Belonging, Growth)
    # ----------------------------------------------------------------------
    df["purpose"] = df["culture_values"]
    df["leadership"] = df["senior_mgmt"]
    df["belonging"] = df[["work_life_balance", "diversity_inclusion"]].mean(axis=1, skipna=True)
    df["growth"] = df["career_opp"]
    df["mwq_proxy"] = df[["purpose", "leadership", "belonging", "growth"]].mean(axis=1, skipna=True)
    logger.info("Calculated MWQ proxy (Purpose, Leadership, Belonging, Growth)")
    
    # ----------------------------------------------------------------------
    # TEXT SIGNALS: Net sentiment proxy from review word counts and polarity
    # ----------------------------------------------------------------------
    text_columns = ['pros', 'cons']
    if all(col in df.columns for col in text_columns):
        analyzer = SentimentIntensityAnalyzer()
        df['pros_length'] = df['pros'].fillna('').str.len()
        df['cons_length'] = df['cons'].fillna('').str.len()
        df['engagement_signal'] = df['pros_length'] - df['cons_length']
        df['pros_sentiment'] = df['pros'].fillna('').apply(lambda x: analyzer.polarity_scores(x)["compound"])
        df['cons_sentiment'] = df['cons'].fillna('').apply(lambda x: analyzer.polarity_scores(x)["compound"])
        df['net_sentiment'] = df['pros_sentiment'] - df['cons_sentiment']
        logger.info("Added text engagement signals")
    else:
        logger.info("No pros/cons columns found - skipping text features")
    
    # ----------------------------------------------------------------------
    # FINAL VALIDATION: Ensure engineered features are present
    # ----------------------------------------------------------------------
    engineered_features = [
        'mwq_proxy', 'belonging_score', 'engagement_signal', 'net_sentiment',
        'belonging_incomplete', 'belonging_imputed'
    ]
    
    present_features = [f for f in engineered_features if f in df.columns]
    
    logger.info("Feature Engineering Summary:")
    logger.info(f"  Final shape: {df.shape}")
    logger.info(f"  Engineered features: {present_features}")
    
    return df


def validate_processed_data(df: pd.DataFrame) -> None:
    """Validate engineered dataset schema and critical ranges."""
    schema = DataFrameSchema(
        {
            "overall_rating": Column(float, coerce=True, nullable=False, checks=Check.in_range(1.0, 5.0)),
            "culture_values": Column(float, coerce=True, nullable=True, checks=Check.in_range(1.0, 5.0)),
            "belonging_score": Column(float, coerce=True, nullable=False, checks=Check.in_range(1.0, 5.0)),
            "career_opp": Column(float, coerce=True, nullable=True, checks=Check.in_range(1.0, 5.0)),
            "mwq_proxy": Column(float, coerce=True, nullable=False, checks=Check.in_range(1.0, 5.0)),
        },
        strict=False,
    )
    schema.validate(df, lazy=True)


def generate_data_quality_report(df: pd.DataFrame, output_path: Path) -> None:
    """Generate a lightweight data quality report as JSON."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    report = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_rates": {col: float(df[col].isna().mean()) for col in df.columns},
        "numeric_ranges": {
            col: {"min": float(df[col].min()), "max": float(df[col].max())} for col in numeric_cols
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Data quality report saved: {output_path}")


def main():
    """Main execution pipeline with error handling"""
    input_path = RAW_DATA_DIR / RAW_DATA_FILENAME
    output_path = PROCESSED_DATA_DIR / PROCESSED_DATA_FILENAME
    dated_output_path = PROCESSED_DATA_DIR / f"culture_intelligence_v1_{date.today().strftime('%Y%m%d')}.parquet"
    report_path = PROJECT_ROOT / "artifacts" / f"data_quality_report_{date.today().strftime('%Y%m%d')}.json"
    
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
        
        # 4. Validate processed data
        validate_processed_data(df_processed)

        # 5. Save to Parquet (fast, compressed, preserves dtypes)
        logger.info(f"Saving to: {output_path.name}")
        df_processed.to_parquet(
            output_path,
            index=False,
            compression="snappy"
        )

        # Also write a versioned copy for reproducibility
        df_processed.to_parquet(
            dated_output_path,
            index=False,
            compression="snappy"
        )

        # 6. Write data quality report
        generate_data_quality_report(df_processed, report_path)
        
        logger.info("Pipeline completed successfully")
        logger.info(f"Output: {len(df_processed):,} rows, {df_processed.shape[1]} columns")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
