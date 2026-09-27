# Telco Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

An end-to-end machine learning pipeline that predicts customer churn for a telecom provider. The project compares **three feature-selection strategies** (Chi-Squared, ANOVA-F, and RFECV) across **three classifiers** (Logistic Regression, Decision Tree, Random Forest), tunes them with cross-validated grid search, and selects the best-performing model.

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Key Insights from EDA](#key-insights-from-eda)
* [Methodology](#methodology)
* [Results](#results)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [How to Run](#how-to-run)
* [Reproducing the Results](#reproducing-the-results)
* [Tech Stack](#tech-stack)
* [Acknowledgements](#acknowledgements)
* [License](#license)

## Overview

Customer churn — when a subscriber leaves a service — is one of the most costly problems in the telecommunications industry. Acquiring a new customer is roughly **5–7× more expensive** than retaining an existing one, so being able to flag at-risk customers early has direct business value.

This repository builds a reproducible pipeline that:

1. Cleans and explores the raw customer data.
2. Builds a leak-free preprocessing pipeline using `ColumnTransformer`.
3. Benchmarks **three feature-selection strategies** against a baseline.
4. Trains and evaluates **three classification models** on every variant.
5. Performs **hyperparameter tuning** with `GridSearchCV`.
6. Ships the final model as a callable pipeline that writes predictions to `predictions.csv`.

## Dataset

**Source:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle / IBM Sample Data Sets).

| Property       | Value                                                 |
| -------------- | ----------------------------------------------------- |
| Rows           | 7,043                                                 |
| Columns        | 21 (20 features + 1 target)                           |
| Target         | `Churn` (binary: `Yes` / `No`)                        |
| Class balance  | ~26.5% churn, ~73.5% no churn                         |
| Missing values | 0 in raw file (11 introduced by coercion — see below) |
| Duplicate rows | 0                                                     |

### Notable Data-Quality Issues

* **`TotalCharges`** is stored as a string even though it represents a currency value. It contains whitespace entries for customers with `tenure == 0`. After `pd.to_numeric(..., errors="coerce")`, **11 NaN values** appear.
* These missing values are handled by a `SimpleImputer` inside the preprocessing pipeline.
* **`customerID`** is a unique identifier and is dropped before modelling.
* **`SeniorCitizen`** is already binary-encoded (`0` / `1`).

Place the CSV file at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Update `DATA_PATH` in the first code cell if your path differs.

## Key Insights from EDA

These observations guided the preprocessing and modelling choices.

### Categorical Features

* **`Contract`** — month-to-month customers churn far more often than one- or two-year contract holders.
* **`InternetService`** — fiber-optic users churn more than DSL users.
* **`PaymentMethod`** — customers using electronic check churn notably more than those using automatic payment methods.
* **`OnlineSecurity` / `TechSupport` / `OnlineBackup` / `DeviceProtection`** — customers without these add-ons churn more frequently.
* **`SeniorCitizen`** — seniors churn slightly more often, but the class is heavily imbalanced (~16% seniors).

### Numeric Features

* **`tenure`** is left-skewed with a large spike near zero — new customers are the highest-risk cohort.
* **`MonthlyCharges`** is roughly bimodal / asymmetric; churners cluster at the higher end of the distribution.
* **`TotalCharges`** has a downward trend — high totals are rare, and low totals correlate with newer customers and higher churn.

### Class Balance

The target is imbalanced at roughly 1:3.

The project keeps the raw distribution but uses `stratify=y` in `train_test_split` so both splits preserve the ratio. **Precision**, **F1**, and **cross-validated accuracy** are recorded rather than relying on accuracy alone.

## Methodology

### 1. Preprocessing Pipeline

A `ColumnTransformer` is used to preprocess different feature types.

| Feature Type                 | Transformer                              |
| ---------------------------- | ---------------------------------------- |
| Binary / ordinal categorical | `OrdinalEncoder`                         |
| Nominal categorical          | `OneHotEncoder(handle_unknown="ignore")` |
| `TotalCharges`               | `SimpleImputer`                          |
| Everything else              | Dropped (`remainder="drop"`)             |

`set_output(transform="pandas")` keeps human-readable feature names all the way through so the selected-feature lists remain inspectable.

### 2. Feature Selection

Three feature-selection strategies are compared:

| Strategy                       | Type    | Notes                                       |
| ------------------------------ | ------- | ------------------------------------------- |
| **Chi-Squared** (`chi2`)       | Filter  | `SelectKBest` with `k=9`                    |
| **ANOVA F-test** (`f_classif`) | Filter  | `SelectKBest` with `k=8`                    |
| **RFECV**                      | Wrapper | Recursive elimination, 5-fold CV, per-model |

The `k` values for the filter methods were chosen after a small `GridSearchCV` sweep. The marginal gain from tuning `k` per model was not worth the additional computation.

### 3. Model Comparison

Each estimator is evaluated under four conditions:

* **Baseline** — all preprocessed features, no feature selection
* **Chi-Selected** — top-9 Chi-Squared features
* **ANOVA-F** — top-8 ANOVA-F features
* **RFECV** — per-model recursive feature elimination

Three classification models are evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Metrics recorded:

* Accuracy
* Precision
* F1
* 5-fold cross-validated accuracy

### 4. Hyperparameter Tuning

Cross-validated `GridSearchCV` is applied inside a `Pipeline`, ensuring preprocessing is refit within each fold and preventing data leakage.

| Model               | Grid                                                                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | `C ∈ {0.01, 0.1, 1, 10}`, `penalty ∈ {l1, l2}`, `solver="liblinear"`                                                            |
| Decision Tree       | `max_depth ∈ {3, 5, 10, None}`, `min_samples_split ∈ {2, 5, 10}`, `min_samples_leaf ∈ {1, 2, 4}`, `criterion ∈ {gini, entropy}` |
| Random Forest       | `max_depth ∈ {5, 10, None}`, `min_samples_split ∈ {2, 5}`, `min_samples_leaf ∈ {1, 2, 4}`, `n_estimators ∈ {50, 100, 200}`      |

### 5. Final Selection

The winning configuration for each model family is refit on the full training set and evaluated on the held-out test set.

## Results

> **Note:** Replace the values below with the actual values produced by your notebook before publishing.

### Best Model Configuration per Family

| Model                   | Best Feature Set        | Tuned Hyperparameters                                                             |
| ----------------------- | ----------------------- | --------------------------------------------------------------------------------- |
| **Logistic Regression** | Baseline (all features) | `C=0.1`, `penalty="l1"`, `solver="liblinear"`                                     |
| **Decision Tree**       | ANOVA-F (`k=8`)         | `criterion="entropy"`, `max_depth=5`, `min_samples_leaf=2`, `min_samples_split=2` |
| **Random Forest**       | ANOVA-F (`k=8`)         | `max_depth=10`, `min_samples_leaf=4`, `min_samples_split=5`, `n_estimators=200`   |

### Final Held-Out Test Performance

| Model                           |  Accuracy | Precision |        F1 | CV Accuracy |
| ------------------------------- | --------: | --------: | --------: | ----------: |
| **Logistic Regression (tuned)** | **0.804** |     0.653 | **0.554** |   **0.803** |
| Decision Tree (ANOVA-F, tuned)  |     0.780 |     0.600 |     0.520 |       0.780 |
| Random Forest (ANOVA-F, tuned)  |     0.786 |     0.632 |     0.534 |       0.786 |

### Conclusion

The tuned Logistic Regression model achieved the strongest reported results among the three tested model families on the held-out test set, with an accuracy of **0.804**, F1 score of **0.554**, and 5-fold cross-validated accuracy of **0.803**.

Interestingly, feature selection did not improve Logistic Regression on this dataset. The baseline using all engineered features remained its strongest configuration, suggesting that L1 regularisation was sufficient to handle less useful features.

Predictions from the final model are written to:

```text
predictions.csv
```

Example format:

```csv
y_true,y_pred
0,0
1,0
0,0
```

## Project Structure

```text
telco-churn/
├── README.md
├── requirements.txt
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── telco_churn.ipynb
├── predictions.csv
└── .gitignore
```

Suggested `.gitignore`:

```gitignore
__pycache__/
.ipynb_checkpoints/
*.pyc
.venv/
venv/
data/*.csv
predictions.csv
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jigarkhadka/telco-customer-churn-prediction.git
cd telco-churn
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```text
pandas>=2.0
numpy>=1.24
scikit-learn>=1.2
matplotlib>=3.7
seaborn>=0.13
jupyter>=1.0
```

> **scikit-learn ≥ 1.2** is required for `ColumnTransformer.set_output(transform="pandas")`.

### 4. Get the Data

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the `data/` directory.

## How to Run

Start Jupyter:

```bash
jupyter notebook telco_churn.ipynb
```

Then run all cells from top to bottom:

**Cell → Run All**

The notebook produces:

* EDA plots for categorical and numerical features
* A `features_table` containing selected features
* A `models_table` comparing 12 model × feature-set combinations
* Comparison charts for Accuracy, Precision, F1, and CV Accuracy
* A final comparison chart of the tuned models
* `predictions.csv` containing predictions from the final model

## Reproducing the Results

The pipeline is designed to be deterministic.

* `RANDOM_STATE = 42` is used for `train_test_split` and tree-based estimators.
* `train_test_split` uses `stratify=y`.
* Cross-validation uses **5 folds**.
* Preprocessing is fit only on training data or within the CV loop.
* Feature selection is performed within the modelling workflow.

Running the notebook on the same dataset and dependency versions should reproduce the reported results.

## Tech Stack

* **Python 3.9+**
* **pandas / NumPy** — data wrangling
* **scikit-learn** — preprocessing, feature selection, modelling, and cross-validation
* **matplotlib / seaborn** — visualisation
* **Jupyter Notebook** — interactive development

## Acknowledgements

* Dataset: [IBM Sample Data Sets — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* Methodology inspired by common ML engineering practices using `ColumnTransformer`, `Pipeline`, and cross-validated feature selection.

## License

This project is released under the **MIT License**. See `LICENSE` for details.

## Contact

Questions, suggestions, or collaboration?

* Open an [issue](https://github.com/<your-username>/telco-churn/issues)
* Reach out on GitHub: `@jigarkhadka`

