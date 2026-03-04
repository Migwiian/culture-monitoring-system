#!/usr/bin/env python
"""
Module: audit_culture.py
Description: EDA validation of Voluntās MWQ proxy
Answers: "Is our engineered MWQ proxy a good signal or random noise?"
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, PROCESSED_DATA_FILENAME

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_processed_data(data_path: Path) -> pd.DataFrame:
    """Load the processed culture intelligence data"""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df

def validate_mwq_proxy(df: pd.DataFrame) -> dict:
    """
    Core validation: Does mwq_proxy actually predict overall_rating?
    If not, we need to revise our feature engineering strategy.
    """
    # 1. CORRELATION ANALYSIS
    # Calculate the correlation between the MWQ proxy and the Overall Rating
    correlation = df['mwq_proxy'].corr(df['overall_rating'])
    logger.info(f"MWQ proxy vs Overall Rating: r={correlation:.3f}")
    
    # Interpretation: Evaluate the strength of the relationship
    if correlation > 0.7:
        # If correlation is high (above 0.7), the index is a strong predictor
        logger.info("EXCELLENT: Index is a strong predictor")
    elif correlation > 0.5:
        # If correlation is moderate (between 0.5 and 0.7), the index is a good predictor
        logger.info("GOOD: Index is moderately predictive")
    else:
        # If correlation is low (below 0.5), the index needs redesign
        logger.error("WEAK: Index needs redesign")
    
    # 2. RANKING VALIDATION (Top 10 Meaningfulness Leaders)
    # Identify the top 10 firms with the highest MWQ proxy scores
    logger.info("\nMEANINGFULNESS LEADERS (Top 10):")
    top_firms = (
        df.groupby('firm')['mwq_proxy']
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    for rank, (firm, score) in enumerate(top_firms.items(), 1):
        # Log the rank, firm, and MWQ proxy score for each top firm
        logger.info(f"   {rank:2d}. {firm:<30} | Score: {score:.2f}")
    
    # Calculate the burnout risk by subtracting the Belonging Score from the Career Opportunity Score
    df['burnout_risk'] = df['career_opp'] - df['belonging_score']
    
    logger.info("\nHIGHEST BURNOUT RISK (Growth vs Belonging Gap):")
    risk_firms = (
        df.groupby('firm')['burnout_risk']
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    for rank, (firm, gap) in enumerate(risk_firms.items(), 1):
        # Determine the status of each firm based on the burnout risk gap
        status = "CRITICAL" if gap > 1.0 else "WARNING"
        # Log the rank, firm, and burnout risk gap for each high-risk firm
        logger.info(f"   {rank:2d}. {firm:<30} | Gap: {gap:.2f} {status}")
    
    # Return the validation results as a dictionary
    return {
        'correlation': correlation,
        'top_firms': top_firms,
        'risk_firms': risk_firms
    }

def detect_anomalies(df: pd.DataFrame) -> None:
    """
    Detect data quality issues that would poison the model
    """
    logger.info("\n ANOMALY DETECTION:")

    # 1. MISSING VALUE RATES
    # Calculate the proportion of missing values in each column
    # This is useful for identifying features that are sparse or have a high proportion of missing data
    missing_rates = df.isna().mean()
    for col, rate in missing_rates[missing_rates > 0].items():
        logger.info(f"   {col}: {rate:.1%} missing")

    # 2. OUT-OF-RANGE RATINGS
    # Check if the 'overall_rating' column contains any values outside the valid range of 1 to 5
    # This is useful for identifying rows with invalid data that would poison the model
    if 'overall_rating' in df.columns:
        # Get the rows where the 'overall_rating' is outside the valid range
        invalid_ratings = df[~df['overall_rating'].between(1.0, 5.0)]
        if len(invalid_ratings) > 0:
            logger.error(f"{len(invalid_ratings)} rows have invalid ratings")

    # 3. MWQ PROXY OUTLIERS
    # Calculate the first quartile (Q1) and third quartile (Q3) of the 'mwq_proxy' column
    # The interquartile range (IQR) is the difference between the Q3 and Q1
    q1 = df['mwq_proxy'].quantile(0.25)
    q3 = df['mwq_proxy'].quantile(0.75)
    iqr = q3 - q1
    # Get the rows where the 'mwq_proxy' is more than 1.5 times the IQR away from the Q1 or Q3
    # These rows are considered outliers and may need to be removed or transformed
    outliers = df[
        (df['mwq_proxy'] < q1 - 1.5 * iqr) | 
        (df['mwq_proxy'] > q3 + 1.5 * iqr)
    ]
    logger.info(f"MWQ proxy outliers: {len(outliers):,} rows ({len(outliers)/len(df):.1%})")

def generate_feature_report(df: pd.DataFrame) -> None:
    """Generate statistical report for each engineered feature"""
    logger.info("\n FEATURE STATISTICS:")
    
    key_features = [
        'purpose_score', 'belonging_score', 'growth_score',
        'mwq_proxy', 'engagement_signal'
    ]
    
    for feature in key_features:
        if feature in df.columns:
            stats = {
                'mean': df[feature].mean(),
                'std': df[feature].std(),
                'min': df[feature].min(),
                'max': df[feature].max(),
                'missing': df[feature].isna().mean()
            }
            
            logger.info(f"   {feature:<20} | "
                       f"μ={stats['mean']:.2f} σ={stats['std']:.2f} | "
                       f"range=[{stats['min']:.2f}, {stats['max']:.2f}] | "
                       f"missing={stats['missing']:.1%}")

if __name__ == "__main__":
    # Robust path resolution
    DATA_PATH = PROCESSED_DATA_DIR / PROCESSED_DATA_FILENAME
    
    try:
        # 1. Load data
        df = load_processed_data(DATA_PATH)
        
        # 2. Validate the index
        results = validate_mwq_proxy(df)
        
        # 3. Detect anomalies
        detect_anomalies(df)
        
        # 4. Generate feature report
        generate_feature_report(df)
        
        logger.info("\nCulture audit completed!")
        
        # 5. Exit with error if index is weak
        if results['correlation'] < 0.5:
            logger.error("MWQ proxy is too weak - redesign required")
            exit(1)
            
    except Exception as e:
        logger.error(f"Audit failed: {str(e)}")
        raise
