"""Model training & selection (Phases 6-8).

Trains four classifiers inside a shared preprocessing pipeline, compares them
with stratified cross-validation (ROC-AUC) and on a held-out test set, then
persists the best pipeline together with its metadata and metrics.

Run as a script::

    python -m src.train_model
"""
from __future__ import annotations

import json
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import config
from .data_preprocessing import clean_data, load_raw_data, split_features_target
from .evaluate_model import evaluate_classifier
from .feature_engineering import build_preprocessor

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_XGB = False

import joblib

warnings.filterwarnings("ignore", category=UserWarning)


def build_model_zoo(scale_positive_weight: float = 1.0) -> Dict[str, object]:
    """Return the dictionary of untrained estimators to compare."""
    models: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=config.RANDOM_STATE, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            random_state=config.RANDOM_STATE,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
            class_weight="balanced",
        ),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
            scale_pos_weight=scale_positive_weight,
            tree_method="hist",
        )
    return models


def train_and_select(save: bool = True) -> Dict:
    """Full training routine. Returns a results dictionary."""
    config.ensure_directories()

    raw = load_raw_data()
    cleaned = clean_data(raw)
    X, y = split_features_target(cleaned)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    # class imbalance ratio for XGBoost
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight = n_neg / max(n_pos, 1)

    preprocessor = build_preprocessor(X_train)
    models = build_model_zoo(scale_positive_weight=pos_weight)

    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )

    results = {}
    fitted_pipelines = {}
    for name, estimator in models.items():
        pipe = Pipeline(
            steps=[("preprocess", preprocessor), ("model", estimator)]
        )
        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        pipe.fit(X_train, y_train)
        test_metrics = evaluate_classifier(pipe, X_test, y_test)
        test_metrics["cv_roc_auc_mean"] = float(np.mean(cv_scores))
        test_metrics["cv_roc_auc_std"] = float(np.std(cv_scores))
        results[name] = test_metrics
        fitted_pipelines[name] = pipe
        print(
            f"{name:>20}  CV-AUC={np.mean(cv_scores):.4f}  "
            f"Test-AUC={test_metrics['roc_auc']:.4f}  "
            f"F1={test_metrics['f1']:.4f}  Recall={test_metrics['recall']:.4f}"
        )

    # Select best model by cross-validated ROC-AUC (selection on training data).
    best_name = max(results, key=lambda k: results[k]["cv_roc_auc_mean"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nSelected best model: {best_name}")

    if save:
        joblib.dump(best_pipeline, config.BEST_MODEL_FILE)

        with open(config.METRICS_FILE, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

        metadata = {
            "best_model": best_name,
            "feature_columns": list(X.columns),
            "numeric_features": [c for c in config.NUMERIC_FEATURES if c in X.columns],
            "categorical_features": [
                c for c in config.CATEGORICAL_FEATURES if c in X.columns
            ],
            "categorical_options": {
                c: sorted(cleaned[c].dropna().unique().tolist())
                for c in config.CATEGORICAL_FEATURES
                if c in cleaned.columns
            },
            "numeric_ranges": {
                c: {
                    "min": float(cleaned[c].min()),
                    "max": float(cleaned[c].max()),
                    "median": float(cleaned[c].median()),
                }
                for c in config.NUMERIC_FEATURES
                if c in cleaned.columns
            },
            "test_size": config.TEST_SIZE,
            "random_state": config.RANDOM_STATE,
            "has_xgboost": _HAS_XGB,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }
        with open(config.METADATA_FILE, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        print(f"Saved model -> {config.BEST_MODEL_FILE}")
        print(f"Saved metrics -> {config.METRICS_FILE}")
        print(f"Saved metadata -> {config.METADATA_FILE}")

    return {
        "results": results,
        "best_name": best_name,
        "best_pipeline": best_pipeline,
        "splits": (X_train, X_test, y_train, y_test),
    }


if __name__ == "__main__":  # pragma: no cover
    train_and_select(save=True)
