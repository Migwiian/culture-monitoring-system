# src/data/make_dataset.py - Data validation & cleaning
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_raw_data(input_path: str = "data/raw/glassdoor_reviews.csv"):
    """Load and validate the 850K dataset"""
    logger.info(f"Loading data from {input_path}")
    
    # Load sample first to check structure
    df = pd.read_csv(input_path, nrows=1000)
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Shape: {df.shape}")
    
    # Check for required columns (adjust based on your CSV)
    required_cols = ["overall_rating", "culture_values", "work_life_balance"]
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    logger.info("✅ Data validation passed")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer Voluntās features"""
    logger.info("Cleaning data...")
    
    # Drop rows with missing ratings
    df = df.dropna(subset=["overall_rating"])
    
    # Engineer Voluntās "Meaningfulness" pillars
    df["purpose_score"] = df["culture_values"]  # Direct mapping
    df["belonging_score"] = (df["work_life_balance"] + df["senior_management"]) / 2
    df["growth_score"] = df["career_opportunities"]
    
    logger.info(f"Cleaned shape: {df.shape}")
    return df

if __name__ == "__main__":
    # This will be your first test
    df_sample = validate_raw_data()
    df_clean = clean_data(df_sample)
    
    # Save processed data
    output_path = Path("data/processed/reviews_clean.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")