#!/usr/bin/env python
"""
Module: audit_culture.py
Description: EDA validation of Voluntās Meaningfulness Index
Answers: "Is our engineered index a good signal or random noise?"
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

def load_processed_data(data_path: Path) -> pd.DataFrame:
    """Load the processed culture intelligence data"""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df

def validate_voluntas_index(df: pd.DataFrame) -> dict:
    """
    Core validation: Does voluntas_index actually predict overall_rating?
    If not, we need to revise our feature engineering strategy.
    """
    # 1. CORRELATION ANALYSIS
    # Calculate the correlation between the Voluntās Index and the Overall Rating
    correlation = df['voluntas_index'].corr(df['overall_rating'])
    logger.info(f"Voluntās Index vs Overall Rating: r={correlation:.3f}")
    
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
    # Identify the top 10 firms with the highest Voluntās Index scores
    logger.info("\nVOLUNTĀS MEANINGFULNESS LEADERS (Top 10):")
    top_firms = (
        df.groupby('firm')['voluntas_index']
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    for rank, (firm, score) in enumerate(top_firms.items(), 1):
        # Log the rank, firm, and Voluntās Index score for each top firm
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

    # 3. VOLUNTĀS INDEX OUTLIERS
    # Calculate the first quartile (Q1) and third quartile (Q3) of the 'voluntas_index' column
    # The interquartile range (IQR) is the difference between the Q3 and Q1
    q1 = df['voluntas_index'].quantile(0.25)
    q3 = df['voluntas_index'].quantile(0.75)
    iqr = q3 - q1
    # Get the rows where the 'voluntas_index' is more than 1.5 times the IQR away from the Q1 or Q3
    # These rows are considered outliers and may need to be removed or transformed
    outliers = df[
        (df['voluntas_index'] < q1 - 1.5 * iqr) | 
        (df['voluntas_index'] > q3 + 1.5 * iqr)
    ]
    logger.info(f"Voluntās Index outliers: {len(outliers):,} rows ({len(outliers)/len(df):.1%})")

def generate_feature_report(df: pd.DataFrame) -> None:
    """Generate statistical report for each engineered feature"""
    logger.info("\n FEATURE STATISTICS:")
    
    key_features = [
        'purpose_score', 'belonging_score', 'growth_score',
        'voluntas_index', 'engagement_signal'
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
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "culture_intelligence_v1.parquet"
    
    try:
        # 1. Load data
        df = load_processed_data(DATA_PATH)
        
        # 2. Validate the index
        results = validate_voluntas_index(df)
        
        # 3. Detect anomalies
        detect_anomalies(df)
        
        # 4. Generate feature report
        generate_feature_report(df)
        
        logger.info("\nCulture audit completed!")
        
        # 5. Exit with error if index is weak
        if results['correlation'] < 0.5:
            logger.error("Voluntās Index is too weak - redesign required")
            exit(1)
            
    except Exception as e:
        logger.error(f"Audit failed: {str(e)}")
        raise

outliers = df[(df['voluntas_index'] < 1.5) | (df['voluntas_index'] > 4.5)]
print("Outlier firms:", outliers['firm'].value_counts().head())