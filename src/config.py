"""Central project configuration (paths, seeds)."""
from pathlib import Path

# Project root is the parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data lives under src/data in this repo
DATA_DIR = PROJECT_ROOT / "src" / "data"
RAW_DATA_DIR = DATA_DIR
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

RANDOM_SEED = 42

# Default dataset/model filenames
RAW_DATA_FILENAME = "glassdoor_reviews.csv"
PROCESSED_DATA_FILENAME = "culture_intelligence_v1.parquet"
BEST_MODEL_FILENAME = "best_model.bin"
