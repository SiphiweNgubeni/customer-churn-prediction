"""Prediction helpers for individual customers (Phase 10 / FR08, FR09).

Loads the saved pipeline (preprocessing + model together, so predictions are
consistent) and exposes convenience functions used by the Streamlit dashboard
and the test-suite.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config

import joblib


@lru_cache(maxsize=1)
def load_model(path: str | None = None):
    """Load and cache the trained pipeline."""
    path = path or config.BEST_MODEL_FILE
    return joblib.load(path)


@lru_cache(maxsize=1)
def _expected_columns() -> tuple:
    """Return the feature columns the model was trained on.

    Prefers the saved metadata; falls back to the config feature list so the
    function still works if metadata is missing.
    """
    if config.METADATA_FILE.exists():
        meta = json.loads(config.METADATA_FILE.read_text())
        cols = meta.get("feature_columns")
        if cols:
            return tuple(cols)
    return tuple(config.ALL_FEATURES)


def _align_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``frame`` has exactly the columns the model expects.

    Missing columns are added as NaN so the pipeline's imputers can fill them
    (this is what makes partial / invalid input safe to predict on). Extra
    columns are dropped. This satisfies the reliability requirement that the
    system handles incomplete input gracefully rather than raising.
    """
    expected: List[str] = list(_expected_columns())
    aligned = frame.copy()
    for col in expected:
        if col not in aligned.columns:
            aligned[col] = np.nan
    return aligned[expected]


def risk_category(probability: float) -> str:
    """Translate a churn probability into a human-friendly risk band."""
    if probability >= 0.70:
        return "High Risk"
    if probability >= 0.40:
        return "Medium Risk"
    return "Low Risk"


def predict_single(customer: Dict, model=None) -> Dict:
    """Predict churn for one customer supplied as a dict of feature values.

    Returns the predicted class, probability and risk band. The input dict keys
    must match the model's expected feature columns; missing optional columns
    are tolerated because the pipeline imputes them.
    """
    model = model or load_model()
    frame = _align_columns(pd.DataFrame([customer]))
    proba = float(model.predict_proba(frame)[0, 1])
    pred = int(proba >= 0.5)
    return {
        "prediction": pred,
        "prediction_label": "Churn" if pred == 1 else "Stay",
        "probability": proba,
        "risk_category": risk_category(proba),
    }


def predict_batch(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Return a copy of ``df`` with churn probability, class and risk columns."""
    model = model or load_model()
    out = df.copy()
    aligned = _align_columns(df)
    proba = model.predict_proba(aligned)[:, 1]
    out["churn_probability"] = proba
    out["churn_prediction"] = (proba >= 0.5).astype(int)
    out["risk_category"] = [risk_category(p) for p in proba]
    return out
