from pathlib import Path
import sys
from datetime import datetime, timedelta

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.train import create_features, temporal_cv_metrics


def _make_df(n=30):
    base_date = datetime(2020, 1, 1)
    rows = []
    for i in range(n):
        rows.append(
            {
                "overall_rating": 3.0 + (i % 3) * 0.5,
                "culture_values": 3.5 + (i % 2) * 0.5,
                "belonging_score": 3.0 + (i % 4) * 0.25,
                "career_opp": 3.2 + (i % 5) * 0.2,
                "engagement_signal": i - 10,
                "net_sentiment": (i % 5) * 0.1,
                "pros_length": 100 + i,
                "cons_length": 80 + i,
                "date_review": base_date + timedelta(days=i),
            }
        )
    return pd.DataFrame(rows)


def test_create_features_includes_optional():
    df = _make_df(5)
    X, y, feature_cols = create_features(df)
    assert "engagement_signal" in feature_cols
    assert "net_sentiment" in feature_cols
    assert "pros_length" in feature_cols
    assert "cons_length" in feature_cols
    assert X.shape[0] == y.shape[0]


def test_temporal_cv_metrics_keys():
    df = _make_df(30)
    metrics = temporal_cv_metrics(df, {"n_estimators": 5, "max_depth": 2})
    assert "linear_mae_mean" in metrics
    assert "xgb_mae_mean" in metrics
    assert "feature_cols" in metrics
