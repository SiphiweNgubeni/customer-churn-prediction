"""Customer Churn Prediction & Analysis - Streamlit dashboard (Phase 10).

Four sections:
    1. Overview Dashboard
    2. Customer Analysis
    3. Churn Prediction (individual risk form)
    4. Model Performance

Run with:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the project root importable so ``src`` works when Streamlit changes cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.data_preprocessing import clean_data, load_raw_data  # noqa: E402
from src.predict import predict_single, risk_category  # noqa: E402

st.set_page_config(page_title="Customer Churn Analysis", page_icon="📉",
                   layout="wide")

GREEN = "#2a9d8f"
RED = "#e76f51"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    df = load_raw_data()
    return clean_data(df)


@st.cache_data(show_spinner=False)
def get_metrics() -> dict:
    if config.METRICS_FILE.exists():
        return json.loads(config.METRICS_FILE.read_text())
    return {}


@st.cache_data(show_spinner=False)
def get_metadata() -> dict:
    if config.METADATA_FILE.exists():
        return json.loads(config.METADATA_FILE.read_text())
    return {}


@st.cache_data(show_spinner=False)
def get_analysis_summary() -> dict:
    path = config.REPORTS_DIR / "analysis_summary.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


@st.cache_resource(show_spinner=False)
def get_model():
    if config.BEST_MODEL_FILE.exists():
        from src.predict import load_model
        return load_model()
    return None


def viz(name: str):
    """Return path to a saved visualisation if it exists."""
    p = config.VISUALS_DIR / name
    return str(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def section_overview(df: pd.DataFrame):
    st.header("Overview Dashboard")
    total = len(df)
    churned = int((df["Churn"] == "Yes").sum())
    churn_rate = churned / total * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churned Customers", f"{churned:,}")
    c3.metric("Churn Rate", f"{churn_rate:.1f}%")
    c4.metric("Retained", f"{total - churned:,}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Distribution")
        if viz("churn_distribution.png"):
            st.image(viz("churn_distribution.png"), use_container_width=True)
        else:
            st.bar_chart(df["Churn"].value_counts())
    with col2:
        st.subheader("Churn by Contract Type")
        if viz("churn_by_contract.png"):
            st.image(viz("churn_by_contract.png"), use_container_width=True)

    st.subheader("Churn rate across customer groups")
    summary = get_analysis_summary()
    rates = summary.get("churn_rate_by_contract")
    if rates:
        st.bar_chart(pd.Series(rates, name="churn rate"))
    else:
        grp = (df.assign(flag=df["Churn"].map({"No": 0, "Yes": 1}))
                 .groupby("Contract")["flag"].mean())
        st.bar_chart(grp)


def section_customer_analysis(df: pd.DataFrame):
    st.header("Customer Analysis")
    st.caption("Explore churn patterns across the main customer attributes. "
               "Rates are shown per group so large groups are not misleading.")

    dimension = st.selectbox(
        "Choose an attribute to analyse",
        ["Contract", "PaymentMethod", "InternetService", "TechSupport",
         "OnlineSecurity", "PaperlessBilling", "SeniorCitizen"],
    )

    tmp = df.assign(flag=df["Churn"].map({"No": 0, "Yes": 1}))
    rate = tmp.groupby(dimension)["flag"].mean().sort_values(ascending=False)
    counts = df.groupby([dimension, "Churn"]).size().unstack(fill_value=0)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Churn rate by {dimension}")
        st.bar_chart(rate)
    with col2:
        st.subheader(f"Customer counts by {dimension}")
        st.bar_chart(counts)

    st.divider()
    st.subheader("Numeric attributes by churn status")
    num_col = st.selectbox("Numeric attribute",
                           ["tenure", "MonthlyCharges", "TotalCharges"])
    plot_map = {
        "tenure": "tenure_by_churn.png",
        "MonthlyCharges": "monthly_charges_by_churn.png",
    }
    if num_col in plot_map and viz(plot_map[num_col]):
        st.image(viz(plot_map[num_col]), width=480)
    d = df.copy()
    d[num_col] = pd.to_numeric(d[num_col], errors="coerce")
    st.dataframe(d.groupby("Churn")[num_col]
                 .agg(["mean", "median", "std", "min", "max"]).round(2))


def section_prediction(df: pd.DataFrame):
    st.header("Churn Prediction")
    model = get_model()
    meta = get_metadata()
    if model is None:
        st.warning("No trained model found. Run `python -m src.train_model` "
                   "first to enable predictions.")
        return

    st.caption("Enter a customer's details to estimate their churn risk.")
    cat_options = meta.get("categorical_options", {})
    num_ranges = meta.get("numeric_ranges", {})

    def opts(col, fallback):
        return cat_options.get(col, fallback)

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", opts("gender", ["Female", "Male"]))
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", opts("Partner", ["Yes", "No"]))
            dependents = st.selectbox("Dependents", opts("Dependents", ["Yes", "No"]))
            phone = st.selectbox("Phone Service", opts("PhoneService", ["Yes", "No"]))
            multiple = st.selectbox("Multiple Lines",
                                    opts("MultipleLines",
                                         ["Yes", "No", "No phone service"]))
        with c2:
            internet = st.selectbox("Internet Service",
                                    opts("InternetService",
                                         ["DSL", "Fiber optic", "No"]))
            security = st.selectbox("Online Security",
                                    opts("OnlineSecurity",
                                         ["Yes", "No", "No internet service"]))
            backup = st.selectbox("Online Backup",
                                  opts("OnlineBackup",
                                       ["Yes", "No", "No internet service"]))
            protection = st.selectbox("Device Protection",
                                      opts("DeviceProtection",
                                           ["Yes", "No", "No internet service"]))
            support = st.selectbox("Tech Support",
                                   opts("TechSupport",
                                        ["Yes", "No", "No internet service"]))
            tv = st.selectbox("Streaming TV",
                              opts("StreamingTV",
                                   ["Yes", "No", "No internet service"]))
        with c3:
            movies = st.selectbox("Streaming Movies",
                                  opts("StreamingMovies",
                                       ["Yes", "No", "No internet service"]))
            contract = st.selectbox("Contract",
                                    opts("Contract",
                                         ["Month-to-month", "One year", "Two year"]))
            paperless = st.selectbox("Paperless Billing",
                                     opts("PaperlessBilling", ["Yes", "No"]))
            payment = st.selectbox("Payment Method",
                                   opts("PaymentMethod",
                                        ["Electronic check", "Mailed check",
                                         "Bank transfer (automatic)",
                                         "Credit card (automatic)"]))
            tenure = st.slider("Tenure (months)", 0,
                               int(num_ranges.get("tenure", {}).get("max", 72)), 12)
            monthly = st.slider("Monthly Charges", 0.0,
                                float(num_ranges.get("MonthlyCharges", {})
                                      .get("max", 120.0)), 70.0)
            total = st.number_input("Total Charges", min_value=0.0,
                                    value=float(round(monthly * max(tenure, 1), 2)))

        submitted = st.form_submit_button("Predict Churn", type="primary")

    if submitted:
        customer = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiple, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": support,
            "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }
        result = predict_single(customer, model=model)
        prob = result["probability"]

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", result["prediction_label"])
        m2.metric("Churn Probability", f"{prob*100:.1f}%")
        m3.metric("Risk Category", result["risk_category"])

        band = result["risk_category"]
        if band == "High Risk":
            st.error(f"High churn risk ({prob*100:.1f}%). Consider proactive "
                     "retention outreach.")
        elif band == "Medium Risk":
            st.warning(f"Medium churn risk ({prob*100:.1f}%). Monitor this customer.")
        else:
            st.success(f"Low churn risk ({prob*100:.1f}%).")

        st.caption("Key contributing factors (heuristic, based on known churn "
                   "drivers in this dataset):")
        factors = []
        if contract == "Month-to-month":
            factors.append("Month-to-month contract raises churn risk.")
        if tenure < 12:
            factors.append("Short tenure (< 12 months) is associated with churn.")
        if payment == "Electronic check":
            factors.append("Electronic-check payment correlates with higher churn.")
        if internet == "Fiber optic":
            factors.append("Fiber-optic customers churn more often in this dataset.")
        if support in ("No", "No internet service"):
            factors.append("Lack of tech support is associated with churn.")
        if not factors:
            factors.append("Profile resembles historically retained customers.")
        for f in factors:
            st.write(f"- {f}")


def section_model_performance():
    st.header("Model Performance")
    metrics = get_metrics()
    meta = get_metadata()
    if not metrics:
        st.warning("No metrics found. Run `python -m src.train_model` first.")
        return

    best = meta.get("best_model", "")
    st.caption(f"Selected model: **{best}** (chosen by cross-validated ROC-AUC).")

    rows = []
    for name, m in metrics.items():
        rows.append({
            "Model": name,
            "Accuracy": round(m["accuracy"], 3),
            "Precision": round(m["precision"], 3),
            "Recall": round(m["recall"], 3),
            "F1": round(m["f1"], 3),
            "ROC-AUC": round(m["roc_auc"], 3),
            "PR-AUC": round(m["pr_auc"], 3),
            "CV ROC-AUC": round(m.get("cv_roc_auc_mean", float("nan")), 3),
        })
    table = pd.DataFrame(rows).set_index("Model")
    st.dataframe(table, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if viz("confusion_matrix.png"):
            st.subheader("Confusion Matrix")
            st.image(viz("confusion_matrix.png"), use_container_width=True)
        if viz("roc_curve.png"):
            st.subheader("ROC Curve")
            st.image(viz("roc_curve.png"), use_container_width=True)
    with col2:
        if viz("pr_curve.png"):
            st.subheader("Precision-Recall Curve")
            st.image(viz("pr_curve.png"), use_container_width=True)
        if viz("feature_importance.png"):
            st.subheader("Feature Importance")
            st.image(viz("feature_importance.png"), use_container_width=True)

    if viz("shap_summary.png"):
        st.subheader("SHAP Summary (Model Interpretability)")
        st.image(viz("shap_summary.png"), use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title("📉 Customer Churn Prediction & Analysis System")
    df = get_data()

    section = st.sidebar.radio(
        "Navigation",
        ["Overview Dashboard", "Customer Analysis", "Churn Prediction",
         "Model Performance"],
    )
    st.sidebar.divider()
    st.sidebar.caption("IBM Telco Customer Churn dataset")
    st.sidebar.caption(f"{len(df):,} customers · {df.shape[1]} columns")

    if section == "Overview Dashboard":
        section_overview(df)
    elif section == "Customer Analysis":
        section_customer_analysis(df)
    elif section == "Churn Prediction":
        section_prediction(df)
    else:
        section_model_performance()


if __name__ == "__main__":
    main()
