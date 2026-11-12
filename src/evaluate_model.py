"""Model evaluation utilities (Phase 8).

Provides a single ``evaluate_classifier`` that returns the full metric suite the
project requires (accuracy, precision, recall, F1, ROC-AUC, PR-AUC) plus helpers
to produce the evaluation plots saved under ``reports/visualisations``.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _positive_scores(model, X):
    """Return probability / score for the positive class."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # scale to 0-1 for a usable score
        return (scores - scores.min()) / (np.ptp(scores) + 1e-9)
    return model.predict(X)


def evaluate_classifier(model, X_test, y_test, threshold: float = 0.5) -> Dict:
    """Compute the standard classification metric suite on a test set."""
    y_scores = _positive_scores(model, X_test)
    y_pred = (y_scores >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_scores)),
        "pr_auc": float(average_precision_score(y_test, y_scores)),
        "confusion_matrix": cm.tolist(),
        "threshold": float(threshold),
    }


def roc_points(model, X_test, y_test):
    y_scores = _positive_scores(model, X_test)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    return fpr, tpr, roc_auc_score(y_test, y_scores)


def pr_points(model, X_test, y_test):
    y_scores = _positive_scores(model, X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    return precision, recall, average_precision_score(y_test, y_scores)
