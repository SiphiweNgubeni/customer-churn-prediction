# Customer Churn Prediction & Analysis System

An end-to-end Data Science project that uses statistical analysis and Machine
Learning to predict customer churn and identify retention opportunities, built
on the IBM Telco Customer Churn dataset. It ships with reproducible analysis
notebooks, a modular training pipeline, four compared models, SHAP
interpretability, and an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Dataset](#4-dataset)
5. [Technologies Used](#5-technologies-used)
6. [Project Structure](#6-project-structure)
7. [Installation](#7-installation)
8. [How to Run](#8-how-to-run)
9. [Methodology](#9-methodology)
10. [Exploratory Data Analysis](#10-exploratory-data-analysis)
11. [Statistical Analysis](#11-statistical-analysis)
12. [Machine Learning Results](#12-machine-learning-results)
13. [Model Interpretability](#13-model-interpretability)
14. [Dashboard](#14-dashboard)
15. [Deployment](#15-deployment)
16. [Limitations](#16-limitations)
17. [Future Improvements](#17-future-improvements)

---

## 1. Project Overview

Customer churn is when customers stop using a company's products or services.
For subscription businesses (telecoms, banking, insurance), retaining existing
customers is far cheaper than acquiring new ones. This system analyses
historical customer data, surfaces the factors associated with churn, and
predicts the probability that an individual customer will leave so the business
can act before the customer cancels.

## 2. Problem Statement

Businesses often only discover that a customer is dissatisfied after they have
already cancelled. This project addresses that by learning churn patterns from
historical data and providing both statistical insight and per-customer risk
estimates to support retention decisions.

## 3. Objectives

1. Obtain a real-world customer churn dataset.
2. Explore and understand the dataset structure.
3. Clean and preprocess the data.
4. Perform Exploratory Data Analysis (EDA).
5. Apply statistical methods to investigate churn patterns.
6. Develop and compare multiple ML classification models.
7. Evaluate models with appropriate classification metrics.
8. Identify the features that drive churn predictions.
9. Build an interactive dashboard.
10. Deploy the application.

## 4. Dataset

- **Source:** IBM Telco Customer Churn dataset
  ([Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn),
  [IBM repo](https://github.com/IBM/telco-customer-churn-on-icp4d)).
- **Size:** 7,043 customers, 21 columns.
- **Target:** `Churn` (Yes / No) — a binary classification problem.
- **Class balance:** 26.5% churned (1,869) vs 73.5% retained (5,174).
- **Location in this repo:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

**Key data-quality finding:** `TotalCharges` contains 11 blank values that
correspond to brand-new customers with `tenure == 0`. These are filled with `0`
(no billing cycle has completed) rather than the column mean, which would
distort the distribution. This decision is implemented in
`src/data_preprocessing.py` (`clean_data`).

## 5. Technologies Used

| Area | Tools |
|---|---|
| Language | Python 3.13 |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Statistics | scipy, statsmodels |
| Machine Learning | scikit-learn, xgboost |
| Interpretability | shap |
| App / Dashboard | streamlit |
| Persistence | joblib |
| Notebooks | jupyter, ipykernel |

> **Note on Python version:** Python 3.13 is used because the broader
> scientific-Python stack has prebuilt wheels for it. Newer releases (3.15) were
> too new at build time and lacked compatible wheels.

## 6. Project Structure

```text
customer-churn-prediction/
├── data/
│   ├── raw/                     # original dataset (CSV)
│   └── processed/               # cleaned dataset (generated)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_statistical_analysis.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── 07_model_interpretability.ipynb
├── src/
│   ├── config.py                # paths, column groups, random seeds
│   ├── data_preprocessing.py    # load, quality report, clean, split
│   ├── feature_engineering.py   # ColumnTransformer preprocessing pipeline
│   ├── statistical_analysis.py  # chi-square + Cramer's V, group tests
│   ├── train_model.py           # train & select the 4 models
│   ├── evaluate_model.py        # metric suite + ROC/PR helpers
│   ├── generate_reports.py      # produce all plots + analysis_summary.json
│   └── predict.py               # load model, single/batch prediction
├── models/                      # best_model.joblib, metrics, metadata (generated)
├── dashboard/
│   └── app.py                   # Streamlit app (4 sections)
├── reports/
│   ├── analysis_summary.json    # statistical findings (generated)
│   └── visualisations/          # generated PNGs
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 7. Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Create and activate a virtual environment (Python 3.13 recommended)
py -3.13 -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies (longer timeout helps on slow connections)
python -m pip install --timeout 120 --retries 10 -r requirements.txt
```

## 8. How to Run

```bash
# Clean the data and generate all EDA / statistical / evaluation artifacts,
# then train the four models and save the best one:
python -m src.generate_reports

# (Or train only, without regenerating every plot:)
python -m src.train_model

# Run the tests:
python tests/test_preprocessing.py      # or: python -m pytest tests/ -v

# Launch the dashboard:
streamlit run dashboard/app.py
```

The dashboard opens at `http://localhost:8501`. To explore the notebooks in VS
Code, open any file in `notebooks/` and select the `venv` kernel.

## 9. Methodology

The project follows a structured Data Science lifecycle: data collection →
quality assessment → cleaning → EDA → statistical testing → feature engineering
→ model development → evaluation → interpretability → dashboard → deployment.
Preprocessing (imputation, one-hot encoding, scaling) is fitted **inside** a
scikit-learn `Pipeline` on the training split only, so no test-set information
leaks into training. All randomness is seeded (`random_state=42`) for
reproducibility.

## 10. Exploratory Data Analysis

Generated to `reports/visualisations/`: churn distribution, churn by contract
type, tenure and monthly charges by churn status, churn by payment method and
internet service, and a numeric correlation heatmap. Churn is compared as a
**rate within each group**, not raw counts, so larger groups are not misread as
higher-churn groups.

Headline EDA finding — churn rate by contract type:

| Contract | Churn rate |
|---|---|
| Month-to-month | 42.7% |
| One year | 11.3% |
| Two year | 2.8% |

## 11. Statistical Analysis

- **Descriptive statistics** for charges and tenure.
- **Chi-square test of independence** between categorical attributes and churn,
  reported with **Cramer's V** effect size so practical significance is
  separated from statistical significance.
- **Numeric group comparison** (Student / Welch t-test or Mann-Whitney U,
  chosen automatically from normality and variance checks) for tenure and
  charges. All three numeric comparisons used Mann-Whitney U (distributions were
  non-normal) and were significant at the 5% level.

Chi-square results (all significant, p < 0.001):

| Feature | Cramer's V | Association |
|---|---|---|
| Contract | 0.41 | Strongest driver |
| Online Security | 0.35 | Moderate |
| Tech Support | 0.34 | Moderate |
| Internet Service | 0.32 | Moderate |
| Payment Method | 0.30 | Moderate |

Numeric comparison (churned vs retained, means):

| Feature | Churned | Retained |
|---|---|---|
| Tenure (months) | 18.0 | 37.6 |
| Monthly Charges | 74.44 | 61.27 |
| Total Charges | 1,531.80 | 2,549.91 |

Full results are written to `reports/analysis_summary.json`.

## 12. Machine Learning Results

Four models are trained and compared: **Logistic Regression**, **Decision
Tree**, **Random Forest**, and **XGBoost**. Each is wrapped in the shared
preprocessing pipeline, uses balanced class weights to handle the imbalance,
and is evaluated with stratified 5-fold cross-validation (ROC-AUC) plus a
held-out 20% test set. The best model (by cross-validated ROC-AUC) is persisted
to `models/best_model.joblib`.

Test-set performance:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | CV ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.738 | 0.504 | 0.783 | 0.614 | **0.842** | 0.633 | **0.846** |
| Decision Tree | 0.740 | 0.506 | 0.791 | 0.617 | 0.834 | 0.623 | 0.825 |
| Random Forest | 0.762 | 0.535 | 0.778 | 0.634 | 0.841 | 0.646 | 0.845 |
| XGBoost | 0.754 | 0.525 | 0.765 | 0.622 | 0.838 | 0.651 | 0.842 |

**Selected model: Logistic Regression** — highest cross-validated ROC-AUC
(0.846) while remaining the most interpretable. Recall is deliberately
prioritised (balanced class weights) because the business goal is to catch
at-risk customers; the classification threshold can be tuned on validation data
to trade precision against recall as retention budgets dictate.

> These values were produced by running the pipeline on the dataset in this
> repo and are stored in `models/model_metrics.json`. Re-run
> `python -m src.generate_reports` to regenerate them.

## 13. Model Interpretability

SHAP explains the selected model via a global summary plot
(`reports/visualisations/shap_summary.png`) alongside coefficient-based feature
importance (`feature_importance.png`). Contract type, tenure, internet service
type, and payment method are consistently the most influential features.
Feature importance indicates **association, not causation**.

## 14. Dashboard

A four-section Streamlit app (`dashboard/app.py`):

1. **Overview Dashboard** — totals, churn rate, distribution, churn across groups.
2. **Customer Analysis** — interactive exploration by contract, payment, tenure,
   charges, and internet service.
3. **Churn Prediction** — an input form returning predicted class, probability,
   risk band, and contributing factors.
4. **Model Performance** — model comparison table, confusion matrix, ROC and PR
   curves, feature importance, and SHAP summary.

> Add dashboard screenshots here once you have run the app (drag images into
> this section, e.g. `![Overview](reports/visualisations/churn_distribution.png)`).

## 15. Deployment

Recommended: **Streamlit Community Cloud**.

1. Push the project to GitHub (ensure `requirements.txt` is present).
2. Set the app entry point to `dashboard/app.py`.
3. Connect the repo on Streamlit Community Cloud and deploy.
4. Do not upload private credentials or sensitive customer data.

> Add the public app link here after deployment.

## 16. Limitations

- Trained on a single historical snapshot; no temporal / longitudinal signal.
- The "contributing factors" shown in the prediction form are heuristic
  explanations of known dataset drivers, not per-row SHAP values.
- The dataset is relatively small and class-imbalanced (~26% churn).
- Associations found are not evidence of causation.

## 17. Future Improvements

- Integrate a PostgreSQL database for customer records.
- Add automated model retraining and monitoring.
- Customer segmentation via clustering and lifetime-value estimation.
- Expose predictions through a FastAPI service.
- Per-customer SHAP explanations rendered live in the dashboard.

---

## License

Released under the MIT License. The IBM Telco Customer Churn dataset is provided
by IBM for public use.
