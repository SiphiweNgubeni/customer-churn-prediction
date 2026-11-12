"""Central configuration for the Customer Churn Prediction & Analysis System.

Keeping paths, column groups and constants in one place makes the pipeline
reproducible and easy to maintain (see Non-Functional Requirements).
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Project root = parent of the ``src`` directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
VISUALS_DIR = REPORTS_DIR / "visualisations"

RAW_DATA_FILE = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_customer_data.csv"

BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"
METRICS_FILE = MODELS_DIR / "model_metrics.json"
METADATA_FILE = MODELS_DIR / "model_metadata.json"

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
TARGET = "Churn"
ID_COLUMN = "customerID"

# Numerical predictors.
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

# Categorical predictors (everything else that is not the id / target / numeric).
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def ensure_directories() -> None:
    """Create all output directories if they do not already exist."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        VISUALS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
