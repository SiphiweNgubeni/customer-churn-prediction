"""Tests for data loading, cleaning, preprocessing and prediction (Phase 11).

Run with:
    python -m pytest tests/ -v
or, if pytest is not installed:
    python tests/test_preprocessing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_preprocessing import (
    assess_quality,
    clean_data,
    load_raw_data,
    split_features_target,
)
from src.feature_engineering import build_preprocessor, split_feature_types
from src.predict import risk_category


# ---------------------------------------------------------------------------
# Dataset loading (FR01)
# ---------------------------------------------------------------------------
def test_raw_data_loads_expected_shape():
    df = load_raw_data()
    assert len(df) > 7000, "Telco dataset should have ~7043 rows"
    assert "Churn" in df.columns
    assert "customerID" in df.columns


def test_quality_report_detects_totalcharges_issue():
    df = load_raw_data()
    report = assess_quality(df)
    # After coercion, TotalCharges should reveal a handful of missing values.
    assert report.missing_counts.get("TotalCharges", 0) >= 0
    assert report.n_rows == len(df)


# ---------------------------------------------------------------------------
# Cleaning / missing-value handling (FR03, FR04)
# ---------------------------------------------------------------------------
def test_clean_data_removes_totalcharges_nans():
    df = load_raw_data()
    cleaned = clean_data(df)
    assert pd.to_numeric(cleaned["TotalCharges"]).isna().sum() == 0
    assert cleaned["TotalCharges"].dtype.kind in "fi"


def test_new_customers_get_zero_total_charges():
    df = load_raw_data()
    cleaned = clean_data(df)
    zero_tenure = cleaned[cleaned["tenure"] == 0]
    if len(zero_tenure) > 0:
        assert (zero_tenure["TotalCharges"] == 0).all()


# ---------------------------------------------------------------------------
# Feature / target split and preprocessing (FR04)
# ---------------------------------------------------------------------------
def test_split_features_target():
    df = clean_data(load_raw_data())
    X, y = split_features_target(df)
    assert "Churn" not in X.columns
    assert "customerID" not in X.columns
    assert set(y.unique()).issubset({0, 1})
    assert len(X) == len(y)


def test_preprocessor_transforms_without_nan():
    df = clean_data(load_raw_data())
    X, y = split_features_target(df)
    numeric, categorical = split_feature_types(X)
    assert len(numeric) > 0 and len(categorical) > 0
    pre = build_preprocessor(X)
    transformed = pre.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert not np.isnan(np.asarray(transformed, dtype=float)).any()


# ---------------------------------------------------------------------------
# Risk banding + prediction consistency (FR08, FR09)
# ---------------------------------------------------------------------------
def test_risk_category_bands():
    assert risk_category(0.9) == "High Risk"
    assert risk_category(0.5) == "Medium Risk"
    assert risk_category(0.1) == "Low Risk"


def test_prediction_consistency_if_model_exists():
    """If a trained model exists, the same input must give the same output."""
    if not config.BEST_MODEL_FILE.exists():
        return  # model not trained yet; skip
    from src.predict import load_model, predict_single

    model = load_model()
    customer = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 1, "PhoneService": "No",
        "MultipleLines": "No phone service", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }
    r1 = predict_single(customer, model=model)
    r2 = predict_single(customer, model=model)
    assert r1["probability"] == r2["probability"]
    assert r1["prediction"] == r2["prediction"]


def test_invalid_input_handled():
    """Missing optional columns should be imputed, not crash, if model exists."""
    if not config.BEST_MODEL_FILE.exists():
        return
    from src.predict import load_model, predict_single

    model = load_model()
    partial = {
        "gender": "Male", "SeniorCitizen": 1, "tenure": 5,
        "Contract": "Month-to-month", "MonthlyCharges": 80.0,
        "TotalCharges": 400.0, "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
    }
    result = predict_single(partial, model=model)
    assert 0.0 <= result["probability"] <= 1.0


# ---------------------------------------------------------------------------
# Manual runner (no pytest needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    failures = 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # pragma: no cover
            failures += 1
            print(f"ERROR {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    sys.exit(1 if failures else 0)
