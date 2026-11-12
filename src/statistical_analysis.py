"""Statistical analysis routines (Phase 5).

Implements the descriptive statistics, the chi-square test of independence with
Cramer's V effect size, and a numerical group comparison (t-test / Welch /
Mann-Whitney depending on assumptions). Results are returned as plain dicts so
they can be reused by notebooks, the dashboard and the test-suite.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def descriptive_statistics(df: pd.DataFrame) -> Dict:
    """Core descriptive statistics requested in Phase 5, Analysis 1."""
    frame = df.copy()
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    churn_counts = frame[config.TARGET].value_counts()
    churn_prop = frame[config.TARGET].value_counts(normalize=True)
    return {
        "monthly_charges_mean": float(frame["MonthlyCharges"].mean()),
        "monthly_charges_median": float(frame["MonthlyCharges"].median()),
        "tenure_std": float(frame["tenure"].std()),
        "total_charges_min": float(frame["TotalCharges"].min()),
        "total_charges_max": float(frame["TotalCharges"].max()),
        "churn_counts": {str(k): int(v) for k, v in churn_counts.items()},
        "churn_proportions": {str(k): float(v) for k, v in churn_prop.items()},
    }


def cramers_v(confusion: np.ndarray) -> float:
    """Bias-corrected Cramer's V effect size for a contingency table."""
    chi2 = stats.chi2_contingency(confusion)[0]
    n = confusion.sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def chi_square_test(df: pd.DataFrame, feature: str, target: str = config.TARGET) -> Dict:
    """Chi-square test of independence between ``feature`` and ``target``."""
    table = pd.crosstab(df[feature], df[target])
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    return {
        "feature": feature,
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "cramers_v": cramers_v(table.values),
        "significant_at_0.05": bool(p_value < 0.05),
        "contingency_table": table.to_dict(),
    }


def compare_numeric_by_churn(
    df: pd.DataFrame, column: str, target: str = config.TARGET
) -> Dict:
    """Compare a numeric column between churn / non-churn groups.

    Chooses the test based on a normality check (Shapiro on a capped sample) and
    a variance equality check (Levene): Student t, Welch t, or Mann-Whitney U.
    """
    frame = df.copy()
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    group_yes = frame.loc[frame[target] == "Yes", column].dropna()
    group_no = frame.loc[frame[target] == "No", column].dropna()

    # Normality (sample-capped so Shapiro stays well-behaved on large n).
    def _is_normal(sample: pd.Series) -> bool:
        s = sample.sample(min(len(sample), 500), random_state=config.RANDOM_STATE)
        if len(s) < 3:
            return False
        return stats.shapiro(s).pvalue > 0.05

    normal = _is_normal(group_yes) and _is_normal(group_no)
    equal_var = stats.levene(group_yes, group_no).pvalue > 0.05

    if normal:
        test_name = "Student t-test" if equal_var else "Welch t-test"
        stat, p = stats.ttest_ind(group_yes, group_no, equal_var=equal_var)
    else:
        test_name = "Mann-Whitney U"
        stat, p = stats.mannwhitneyu(group_yes, group_no, alternative="two-sided")

    return {
        "column": column,
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p),
        "mean_churn": float(group_yes.mean()),
        "mean_stay": float(group_no.mean()),
        "significant_at_0.05": bool(p < 0.05),
    }
