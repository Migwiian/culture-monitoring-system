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
import sys
import json
from datetime import date
import mlflow
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_FILENAME,
    MODELS_DIR,
    BEST_MODEL_FILENAME,
    MLRUNS_DIR,
    RANDOM_SEED,
)

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
    base_cols = ["culture_values", "belonging_score", "career_opp"]
    optional_cols = ["engagement_signal", "net_sentiment", "pros_length", "cons_length"]
    feature_cols = base_cols + [c for c in optional_cols if c in df.columns]
    X = df[feature_cols]
    y = df['overall_rating']
    return X, y, feature_cols

def split_temporal(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """Temporal train/validation split (no leakage)."""
    df = df.sort_values('date_review').reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    split_date = df.iloc[split_idx]["date_review"]
    first_val_idx = df[df["date_review"] > split_date].index[0]
    
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


def evaluate_model_mae(X_train, y_train, X_val, y_val, model_kind: str, params: dict | None = None) -> float:
    """Evaluate a model without MLflow side effects (for CV)."""
    if model_kind == "linear":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return mean_absolute_error(y_val, y_pred)

    if model_kind == "xgb":
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)
        model = xgb.XGBRegressor(
            **(params or {}),
            random_state=RANDOM_SEED,
            n_jobs=4,
        )
        model.fit(X_train_imp, y_train, eval_set=[(X_val_imp, y_val)], verbose=False)
        y_pred = model.predict(X_val_imp)
        return mean_absolute_error(y_val, y_pred)

    raise ValueError(f"Unknown model_kind: {model_kind}")


def temporal_cv_metrics(df: pd.DataFrame, params: dict) -> dict:
    """Compute time-series CV MAE for linear and xgboost."""
    df = df.dropna(subset=["date_review"]).sort_values("date_review").reset_index(drop=True)
    X, y, feature_cols = create_features(df)

    tscv = TimeSeriesSplit(n_splits=3)
    linear_mae = []
    xgb_mae = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        linear_mae.append(evaluate_model_mae(X_train, y_train, X_val, y_val, "linear"))
        xgb_mae.append(evaluate_model_mae(X_train, y_train, X_val, y_val, "xgb", params=params))

    return {
        "feature_cols": feature_cols,
        "linear_mae_mean": float(np.mean(linear_mae)),
        "linear_mae_std": float(np.std(linear_mae)),
        "xgb_mae_mean": float(np.mean(xgb_mae)),
        "xgb_mae_std": float(np.std(xgb_mae)),
    }

def main():
    """Execute training pipeline: baseline + 2 XGBoost variants."""
    try:
        mlflow.set_tracking_uri(str(MLRUNS_DIR))
        mlflow.set_experiment("culture-intelligence-v1")
        data_path = PROCESSED_DATA_DIR / PROCESSED_DATA_FILENAME

        data = load_data(data_path)
        if data is None or data.empty:
            raise ValueError("Loaded data is empty")

        # Prefer temporal split when date_review exists; fall back to random split
        if "date_review" in data.columns:
            train_df, val_df = split_temporal(data, test_size=0.2)
            train_features, train_target, feature_cols = create_features(train_df)
            val_features, val_target, _ = create_features(val_df)
        else:
            features, target, feature_cols = create_features(data)
            train_features, val_features, train_target, val_target = train_test_split(
                features, target, test_size=0.2, random_state=RANDOM_SEED
            )

        if train_features is None or train_target is None:
            raise ValueError("Feature extraction resulted in empty data")

        models = {}
        models["linear_baseline"] = train_baseline(train_features, train_target, val_features, val_target)
        models["xgb_v1"] = train_xgboost(
            train_features, train_target, val_features, val_target, {"n_estimators": 100, "max_depth": 4}, "xgb_v1"
        )
        models["xgb_v2"] = train_xgboost(
            train_features, train_target, val_features, val_target, {"n_estimators": 200, "max_depth": 6}, "xgb_v2"
        )

        best_model_name = min(models, key=lambda x: models[x][0])
        best_mae, best_model_object = models[best_model_name]
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / BEST_MODEL_FILENAME
        joblib.dump(best_model_object, model_path)

        # Optional temporal CV report (if date_review exists)
        cv_metrics = None
        if "date_review" in data.columns:
            cv_metrics = temporal_cv_metrics(data, {"n_estimators": 200, "max_depth": 6})

        # Write evaluation report
        eval_report = {
            "date": date.today().isoformat(),
            "features": feature_cols,
            "models": {
                name: {"mae": float(mae)} for name, (mae, _model) in models.items()
            },
            "best_model": best_model_name,
        }
        if cv_metrics:
            eval_report["temporal_cv"] = cv_metrics

        eval_path = PROJECT_ROOT / "artifacts" / f"model_eval_{date.today().strftime('%Y%m%d')}.json"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(json.dumps(eval_report, indent=2))

        # Write a lightweight model card
        card_path = MODELS_DIR / "model_card.txt"
        card_lines = [
            "Voluntas Culture Intelligence Model Card",
            f"Date: {date.today().isoformat()}",
            f"Best model: {best_model_name}",
            f"Features: {', '.join(feature_cols)}",
            f"MAE (best): {best_mae:.4f}",
        ]
        if cv_metrics:
            card_lines.append(
                f"Temporal CV (xgb mean±std): {cv_metrics['xgb_mae_mean']:.4f} ± {cv_metrics['xgb_mae_std']:.4f}"
            )
        card_path.write_text("\n".join(card_lines))
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")

if __name__ == "__main__":
    main()
