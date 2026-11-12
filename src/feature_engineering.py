"""Feature preparation pipeline (Phase 6).

The preprocessing is expressed as a scikit-learn :class:`ColumnTransformer` so
that it can be fitted *inside* a :class:`Pipeline` on the training fold only.
This guarantees no test-set information leaks into training (a core
reproducibility requirement of the project).
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def split_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return ``(numeric_cols, categorical_cols)`` present in ``X``."""
    numeric = [c for c in config.NUMERIC_FEATURES if c in X.columns]
    categorical = [c for c in config.CATEGORICAL_FEATURES if c in X.columns]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """Build the preprocessing ColumnTransformer.

    * Numeric: median imputation + optional standardisation.
    * Categorical: most-frequent imputation + one-hot encoding.

    ``scale_numeric`` can be turned off for tree-based models that do not need
    feature scaling, but leaving it on is harmless for them and keeps the
    pipeline uniform. We keep it configurable for clarity.
    """
    numeric, categorical = split_feature_types(X)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Return the expanded feature names after a preprocessor has been fitted."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover - defensive
        return []
