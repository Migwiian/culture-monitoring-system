#!/usr/bin/env python
"""
Module: train.py
Description: Train multiple models for Voluntās Culture Intelligence with MLflow tracking.
Implements model comparison and basic hyperparameter tuning per capstone requirements.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path: Path) -> pd.DataFrame:
    """Load processed features from data pipeline."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} rows")
    return df

def create_features(df: pd.DataFrame) -> tuple:
    """Create feature matrix X and target y."""
    feature_cols = ['culture_values', 'belonging_score', 'career_opp']
    X = df[feature_cols]
    y = df['overall_rating']
    return X, y

def split_temporal(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """Temporal train/validation split (no leakage)."""
    df = df.sort_values('date_review').reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    split_date = df.iloc[split_idx]['date_review']
    first_val_idx = df[df['date_review'] > split_date].index[0]
    
    train_df = df.iloc[:first_val_idx]
    val_df = df.iloc[first_val_idx:]
    return train_df, val_df

def train_baseline(X_train, y_train, X_val, y_val):
    """Train baseline LinearRegression model."""
    logger.info("Training LinearRegression baseline")
    with mlflow.start_run(run_name="linear_baseline"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(model, "linear_model")
        logger.info(f"Linear MAE: {mae:.3f}")
        return mae, model   

def train_xgboost(X_train, y_train, X_val, y_val, params: dict, run_name: str):
    """Train XGBoost with given hyperparameters."""
    logger.info(f"Training XGBoost with params: {params}")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        
        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)
        
        model = xgb.XGBRegressor(**params, 
                                 callbacks=[xgb.callback.EarlyStopping(rounds=20, save_best=True, maximize=False)], 
                                 random_state=42, 
                                 n_jobs=4
        )
        model.fit(
            X_train_imp, y_train,
            eval_set=[(X_val_imp, y_val)],
            verbose=False
        )
        
        y_pred = model.predict(X_val_imp)
        mae = mean_absolute_error(y_val, y_pred)
        mlflow.log_metric("mae", mae)
        logger.info(f"{run_name} MAE: {mae:.3f}")
        return mae, model

def main():
    """Execute training pipeline: baseline + 2 XGBoost variants."""
    mlflow.set_tracking_uri("http://localhost:5000")
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / "data" / "processed" / "culture_intelligence_v1.parquet"
    
    df = load_data(data_path)
    X, y = create_features(df)
    train_df, val_df = split_temporal(df)
    X_train, y_train = create_features(train_df)
    X_val, y_val = create_features(val_df)
    
    # Train multiple models and log results
    results = {}
    
    results['linear_baseline'] = train_baseline(X_train, y_train, X_val, y_val)
    
    results['xgb_v1'] = train_xgboost(X_train, y_train, X_val, y_val, 
                                     {'n_estimators': 100, 'max_depth': 4}, "xgb_v1")
    
    results['xgb_v2'] = train_xgboost(X_train, y_train, X_val, y_val, 
                                     {'n_estimators': 200, 'max_depth': 6}, "xgb_v2")

    # Compare models to find the best one
    best_model_name = min(results, key=results.get)
    best_mae = results[best_model_name]
    model_path = project_root / "models" / "best_model.bin"
    joblib.dump(best_model_name, model_path)
    logger.info(f"Successfully saved {best_model_name} to {model_path}")

if __name__ == "__main__":
    main()