import json
from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.make_dataset import (
    validate_raw_data,
    clean_data,
    generate_data_quality_report,
)


def test_validate_raw_data_missing_columns(tmp_path):
    df = pd.DataFrame({
        "overall_rating": [4.0, 3.0],
        "culture_values": [4.0, 3.0],
        "work_life_balance": [4.0, 3.0],
        "senior_mgmt": [4.0, 3.0],
        # missing diversity_inclusion
        "career_opp": [4.0, 3.0],
    })
    csv_path = tmp_path / "raw.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        validate_raw_data(csv_path)


def test_clean_data_engineered_features():
    df = pd.DataFrame({
        "overall_rating": [4.0, 3.0, 5.0],
        "culture_values": [4.0, None, 5.0],
        "work_life_balance": [4.0, None, 5.0],
        "senior_mgmt": [4.0, None, 5.0],
        "diversity_inclusion": [4.0, None, 5.0],
        "career_opp": [4.0, 3.0, None],
        "pros": ["good", None, "great"],
        "cons": ["ok", "bad", None],
    })

    out = clean_data(df)

    expected_cols = {
        "belonging_score",
        "belonging_imputed",
        "belonging_incomplete",
        "mwq_proxy",
        "engagement_signal",
        "culture_values_imputed",
        "career_opp_imputed",
    }
    assert expected_cols.issubset(set(out.columns))
    assert out["overall_rating"].isna().sum() == 0


def test_generate_data_quality_report(tmp_path):
    df = pd.DataFrame({
        "overall_rating": [4.0, 3.0],
        "culture_values": [4.0, 3.0],
        "belonging_score": [4.0, 3.0],
        "career_opp": [4.0, 3.0],
        "mwq_proxy": [4.0, 3.0],
    })

    report_path = tmp_path / "report.json"
    generate_data_quality_report(df, report_path)

    payload = json.loads(report_path.read_text())
    assert payload["rows"] == 2
    assert "missing_rates" in payload
    assert "numeric_ranges" in payload
