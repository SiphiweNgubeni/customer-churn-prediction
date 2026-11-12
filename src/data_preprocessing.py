"""Data loading, quality assessment and cleaning (Phases 2 & 3).

This module centralises every decision made about the raw Telco dataset so the
work is reproducible and auditable. Nothing here leaks test-set information:
the imputation / scaling that *could* leak is handled inside the modelling
pipeline (see ``feature_engineering.py``), fitted on training data only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_raw_data(path=None) -> pd.DataFrame:
    """Load the raw Telco churn CSV.

    The file shipped with this project carries a ``.xls`` extension but is in
    fact comma-separated text, so it is read with :func:`pandas.read_csv`.
    """
    path = path or config.RAW_DATA_FILE
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Quality assessment (FR02, FR03)
# ---------------------------------------------------------------------------
@dataclass
class DataQualityReport:
    n_rows: int
    n_cols: int
    missing_counts: Dict[str, int] = field(default_factory=dict)
    missing_percentage: Dict[str, float] = field(default_factory=dict)
    duplicate_rows: int = 0
    duplicate_ids: int = 0
    dtypes: Dict[str, str] = field(default_factory=dict)
    churn_distribution: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Rows: {self.n_rows:,}  Columns: {self.n_cols}",
            f"Duplicate rows: {self.duplicate_rows}",
            f"Duplicate customer IDs: {self.duplicate_ids}",
            "Columns with missing values:",
        ]
        any_missing = False
        for col, count in self.missing_counts.items():
            if count > 0:
                any_missing = True
                pct = self.missing_percentage[col]
                lines.append(f"  - {col}: {count} ({pct:.2f}%)")
        if not any_missing:
            lines.append("  (none)")
        lines.append(f"Churn distribution: {self.churn_distribution}")
        return "\n".join(lines)


def assess_quality(df: pd.DataFrame) -> DataQualityReport:
    """Produce a data-quality report on a *coerced* copy of the frame.

    ``TotalCharges`` is coerced to numeric first because blank strings there are
    the dataset's main hidden quality issue.
    """
    frame = df.copy()
    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")

    missing_counts = frame.isnull().sum()
    missing_pct = (missing_counts / len(frame)) * 100

    duplicate_ids = 0
    if config.ID_COLUMN in frame.columns:
        duplicate_ids = len(frame) - frame[config.ID_COLUMN].nunique()

    churn_dist = {}
    if config.TARGET in frame.columns:
        churn_dist = frame[config.TARGET].value_counts().to_dict()

    return DataQualityReport(
        n_rows=len(frame),
        n_cols=frame.shape[1],
        missing_counts=missing_counts.to_dict(),
        missing_percentage=missing_pct.to_dict(),
        duplicate_rows=int(frame.duplicated().sum()),
        duplicate_ids=int(duplicate_ids),
        dtypes={c: str(t) for c, t in frame.dtypes.items()},
        churn_distribution={str(k): int(v) for k, v in churn_dist.items()},
    )


# ---------------------------------------------------------------------------
# Cleaning (FR04)
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw dataframe.

    Decisions documented here:

    * ``TotalCharges`` is coerced to numeric. The ~11 blank values correspond to
      brand-new customers with ``tenure == 0``. For those rows TotalCharges is
      logically ``0`` (no billing cycle has completed), so we fill with 0 rather
      than the column mean, which would distort the distribution.
    * Whitespace is stripped from object columns.
    * Exact duplicate rows are dropped (there are normally none, but we guard
      against it for robustness).
    * ``customerID`` is kept in the cleaned file for traceability but is excluded
      from the model feature set downstream.
    """
    frame = df.copy()

    # Coerce TotalCharges and reason about the missing values.
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    new_customer_mask = frame["TotalCharges"].isna() & (frame["tenure"] == 0)
    frame.loc[new_customer_mask, "TotalCharges"] = 0.0
    # Any remaining NaN (not explained by zero tenure) falls back to the median.
    if frame["TotalCharges"].isna().any():
        frame["TotalCharges"] = frame["TotalCharges"].fillna(
            frame["TotalCharges"].median()
        )

    # Normalise text columns.
    object_cols = frame.select_dtypes(include="object").columns
    for col in object_cols:
        frame[col] = frame[col].str.strip()

    # Drop exact duplicates.
    frame = frame.drop_duplicates().reset_index(drop=True)

    return frame


def split_features_target(df: pd.DataFrame):
    """Separate predictors ``X`` from the binary target ``y`` (FR04/Phase 6).

    ``Churn`` is mapped Yes->1 / No->0. ``customerID`` is removed from features.
    """
    y = df[config.TARGET].map({"No": 0, "Yes": 1}).astype(int)
    drop_cols = [c for c in (config.TARGET, config.ID_COLUMN) if c in df.columns]
    X = df.drop(columns=drop_cols)
    # Keep only the configured feature columns that are present.
    feature_cols = [c for c in config.ALL_FEATURES if c in X.columns]
    X = X[feature_cols]
    return X, y


def save_cleaned(df: pd.DataFrame, path=None) -> None:
    path = path or config.CLEANED_DATA_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":  # pragma: no cover - manual utility
    config.ensure_directories()
    raw = load_raw_data()
    print("Raw shape:", raw.shape)
    report = assess_quality(raw)
    print(report.summary())
    cleaned = clean_data(raw)
    save_cleaned(cleaned)
    print(f"Cleaned data saved to {config.CLEANED_DATA_FILE}")
