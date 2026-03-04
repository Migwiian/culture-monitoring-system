# src/monitoring/drift_report.py
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float | None:
    """Compute Population Stability Index (PSI) for a numeric column."""
    ref = ref.dropna()
    cur = cur.dropna()
    if ref.empty or cur.empty:
        return None

    quantiles = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return None

    ref_hist, _ = np.histogram(ref, bins=quantiles)
    cur_hist, _ = np.histogram(cur, bins=quantiles)

    ref_pct = ref_hist / max(ref_hist.sum(), 1)
    cur_pct = cur_hist / max(cur_hist.sum(), 1)

    # Avoid zero divisions
    eps = 1e-6
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def generate_drift_report(reference_path: Path, current_path: Path, output_dir: Path) -> Path:
    """Generate a lightweight drift report comparing reference vs. current data."""
    logger.info("Loading reference data: %s", reference_path)
    ref = pd.read_parquet(reference_path)
    logger.info("Loading current data: %s", current_path)
    cur = pd.read_parquet(current_path)

    numeric_cols = ref.select_dtypes(include=[np.number]).columns
    report = {
        "date": date.today().isoformat(),
        "reference_rows": int(len(ref)),
        "current_rows": int(len(cur)),
        "columns": {},
        "alerts": [],
    }

    for col in numeric_cols:
        if col not in cur.columns:
            continue
        psi = _psi(ref[col], cur[col])
        status = "ok"
        if psi is not None and psi > 0.3:
            status = "alert"
            report["alerts"].append({"column": col, "psi": psi, "level": "alert"})
        elif psi is not None and psi > 0.2:
            status = "warning"
            report["alerts"].append({"column": col, "psi": psi, "level": "warning"})

        report["columns"][col] = {
            "reference_mean": float(ref[col].mean()),
            "current_mean": float(cur[col].mean()),
            "reference_std": float(ref[col].std()),
            "current_std": float(cur[col].std()),
            "psi": psi,
            "status": status,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"drift_report_{date.today().strftime('%Y%m%d')}.json"
    json_path.write_text(json.dumps(report, indent=2))

    # Minimal HTML report
    html_path = output_dir / f"drift_report_{date.today().strftime('%Y%m%d')}.html"
    rows = []
    for col, metrics in report["columns"].items():
        rows.append(
            f"<tr><td>{col}</td><td>{metrics['psi']}</td><td>{metrics['status']}</td></tr>"
        )
    html = (
        "<html><head><title>Drift Report</title></head><body>"
        "<h1>Drift Report</h1>"
        "<table border=\"1\" cellpadding=\"6\" cellspacing=\"0\">"
        "<tr><th>Column</th><th>PSI</th><th>Status</th></tr>"
        + "\n".join(rows)
        + "</table></body></html>"
    )
    html_path.write_text(html)

    logger.info("Drift report saved: %s", json_path)
    logger.info("Drift report saved: %s", html_path)
    return json_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    reference = project_root / "src" / "data" / "processed" / "culture_intelligence_v1.parquet"
    current = reference
    output = project_root / "artifacts"
    generate_drift_report(reference, current, output)
