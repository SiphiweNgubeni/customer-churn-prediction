"""Generate EDA, statistical and evaluation visualisations (Phases 4, 8, 9).

Running this as a script writes PNGs to ``reports/visualisations`` and a
``reports/analysis_summary.json`` capturing the statistical findings. The
Streamlit dashboard reads these artifacts so it stays fast and does not need to
recompute everything on each page load.

    python -m src.generate_reports
"""
from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for script use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config
from .data_preprocessing import clean_data, load_raw_data, split_features_target
from .evaluate_model import pr_points, roc_points
from .feature_engineering import build_preprocessor
from .statistical_analysis import (
    chi_square_test,
    compare_numeric_by_churn,
    descriptive_statistics,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def _save(fig, name: str) -> None:
    config.VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.VISUALS_DIR / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# EDA plots (Phase 4)
# ---------------------------------------------------------------------------
def generate_eda_plots(df: pd.DataFrame) -> None:
    print("Generating EDA plots...")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Churn", ax=ax, hue="Churn", legend=False,
                  palette=["#2a9d8f", "#e76f51"])
    ax.set_title("Customer Churn Distribution")
    ax.set_ylabel("Number of Customers")
    _save(fig, "churn_distribution.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="Contract", hue="Churn", ax=ax,
                  palette=["#2a9d8f", "#e76f51"])
    ax.set_title("Churn by Contract Type")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, "churn_by_contract.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="Churn", y="tenure", ax=ax, hue="Churn", legend=False,
                palette=["#2a9d8f", "#e76f51"])
    ax.set_title("Customer Tenure by Churn Status")
    _save(fig, "tenure_by_churn.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax, hue="Churn",
                legend=False, palette=["#2a9d8f", "#e76f51"])
    ax.set_title("Monthly Charges by Churn Status")
    _save(fig, "monthly_charges_by_churn.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x="PaymentMethod", hue="Churn", ax=ax,
                  palette=["#2a9d8f", "#e76f51"])
    ax.set_title("Churn by Payment Method")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "churn_by_payment.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="InternetService", hue="Churn", ax=ax,
                  palette=["#2a9d8f", "#e76f51"])
    ax.set_title("Churn by Internet Service")
    _save(fig, "churn_by_internet.png")

    # Correlation heatmap of numeric features.
    num = df.copy()
    num["TotalCharges"] = pd.to_numeric(num["TotalCharges"], errors="coerce")
    num["ChurnFlag"] = num["Churn"].map({"No": 0, "Yes": 1})
    corr = num[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
                "ChurnFlag"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
    ax.set_title("Numeric Feature Correlations")
    _save(fig, "correlation_heatmap.png")


# ---------------------------------------------------------------------------
# Statistical analysis (Phase 5)
# ---------------------------------------------------------------------------
def run_statistical_analysis(df: pd.DataFrame) -> dict:
    print("Running statistical analysis...")
    results = {"descriptive": descriptive_statistics(df), "chi_square": {}, "numeric": {}}

    for feature in ["Contract", "PaymentMethod", "InternetService",
                    "TechSupport", "OnlineSecurity"]:
        results["chi_square"][feature] = chi_square_test(df, feature)

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        results["numeric"][col] = compare_numeric_by_churn(df, col)

    # churn rate by contract (rates not raw counts, per Phase 4 guidance)
    rate = (df.assign(flag=df["Churn"].map({"No": 0, "Yes": 1}))
              .groupby("Contract")["flag"].mean().sort_values(ascending=False))
    results["churn_rate_by_contract"] = {k: float(v) for k, v in rate.items()}

    return results


# ---------------------------------------------------------------------------
# Evaluation plots (Phase 8) + SHAP (Phase 9)
# ---------------------------------------------------------------------------
def generate_evaluation_plots(pipeline, X_test, y_test, best_name: str) -> None:
    print("Generating evaluation plots...")
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, ax=ax,
                                          cmap="Blues",
                                          display_labels=["Stay", "Churn"])
    ax.set_title(f"Confusion Matrix - {best_name}")
    _save(fig, "confusion_matrix.png")

    fpr, tpr, auc = roc_points(pipeline, X_test, y_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="#e76f51")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {best_name}")
    ax.legend()
    _save(fig, "roc_curve.png")

    prec, rec, ap = pr_points(pipeline, X_test, y_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, label=f"PR-AUC = {ap:.3f}", color="#264653")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve - {best_name}")
    ax.legend()
    _save(fig, "pr_curve.png")


def generate_feature_importance(pipeline, X_train, best_name: str) -> None:
    """Permutation / model feature importance plot."""
    print("Generating feature importance...")
    model = pipeline.named_steps["model"]
    pre = pipeline.named_steps["preprocess"]
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(len(getattr(model, "feature_importances_", [])))]

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()

    if importances is None:
        return

    order = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(x=importances[order], y=[names[i] for i in order], ax=ax,
                color="#2a9d8f")
    ax.set_title(f"Top Feature Importances - {best_name}")
    ax.set_xlabel("Importance")
    _save(fig, "feature_importance.png")


def generate_shap_plots(pipeline, X_train) -> None:
    """SHAP global summary (Phase 9). Degrades gracefully if SHAP absent."""
    try:
        import shap
    except Exception:
        print("  SHAP not available, skipping SHAP plots.")
        return

    print("Generating SHAP summary (this can take a moment)...")
    try:
        pre = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]
        sample = X_train.sample(min(len(X_train), 400),
                                random_state=config.RANDOM_STATE)
        X_trans = pre.transform(sample)
        names = list(pre.get_feature_names_out())

        explainer = shap.Explainer(model, X_trans, feature_names=names)
        shap_values = explainer(X_trans)

        fig = plt.figure()
        shap.summary_plot(shap_values, features=X_trans, feature_names=names,
                          show=False, plot_size=(8, 6))
        _save(plt.gcf(), "shap_summary.png")
    except Exception as exc:  # pragma: no cover - SHAP can be finicky
        print(f"  SHAP plot failed: {exc}")


def main() -> None:
    config.ensure_directories()
    raw = load_raw_data()
    df = clean_data(raw)

    generate_eda_plots(df)
    stats_results = run_statistical_analysis(df)

    with open(config.REPORTS_DIR / "analysis_summary.json", "w", encoding="utf-8") as fh:
        json.dump(stats_results, fh, indent=2)
    print(f"  saved analysis_summary.json")

    # Train (or load) to produce evaluation + importance + SHAP plots.
    from .train_model import train_and_select

    outcome = train_and_select(save=True)
    X_train, X_test, y_train, y_test = outcome["splits"]
    pipeline = outcome["best_pipeline"]
    best_name = outcome["best_name"]

    generate_evaluation_plots(pipeline, X_test, y_test, best_name)
    generate_feature_importance(pipeline, X_train, best_name)
    generate_shap_plots(pipeline, X_train)
    print("All reports generated.")


if __name__ == "__main__":  # pragma: no cover
    main()
