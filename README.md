to be specified
# 🏥 Obesity Risk Estimation — GR 27

> **École Centrale Casablanca** — Machine Learning & Data Science Project  
> Estimation of Obesity Levels Based on Eating Habits and Physical Condition

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?logo=streamlit)
![License](https://img.shields.io/badge/License-CC%20BY%204.0-green)

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Project Structure](#-project-structure)
4. [Installation & Reproducibility](#-installation--reproducibility)
5. [Usage](#-usage)
6. [Exploratory Data Analysis](#-exploratory-data-analysis)
7. [Models & Results](#-models--results)
8. [SHAP Explainability](#-shap-explainability)
9. [Web Application](#-web-application)
10. [Automated Testing & CI](#-automated-testing--ci)
11. [Prompt Engineering Documentation](#-prompt-engineering-documentation)
12. [Required Questions](#-required-questions)
13. [License](#-license)

---

## 🎯 Project Overview

This project builds a **clinical decision-support system** that predicts a patient's obesity risk level (7 classes) from lifestyle and physical-condition features. It includes:

- A complete **ML pipeline** comparing Random Forest, XGBoost, LightGBM, and CatBoost.
- **SHAP explainability** for transparent, interpretable predictions.
- A professional **Streamlit web dashboard** designed as a medical interface.
- Full **automated testing** with GitHub Actions CI.

---

## 📊 Dataset

| Item | Detail |
|---|---|
| **Source** | [UCI ML Repository #544](https://archive.ics.uci.edu/dataset/544) |
| **Instances** | 2 111 |
| **Features** | 16 (eating habits + physical condition) |
| **Target** | `NObeyesdad` — 7 obesity levels |
| **Missing values** | None |
| **Synthetic data** | 77 % generated via SMOTE; 23 % collected from users |

### How the Dataset Is Fetched

The function `fetch_dataset()` in `src/data_processing.py`:

1. Uses the [`ucimlrepo`](https://github.com/uci-ml-repo/ucimlrepo) package to download dataset **544** programmatically.
2. Saves the data to `data/raw/obesity_data.csv` for caching.
3. On subsequent calls, loads directly from the local CSV.

```python
from src.data_processing import fetch_dataset
df = fetch_dataset()  # Downloads on first run, then loads from cache
```

No manual download is required.

---

## 📁 Project Structure

```
obesity-risk-gr27/
├── app/
│   ├── app.py                  # Streamlit web application
│   └── assets/
│       └── ecc_logo.png        # École Centrale Casablanca logo
├── data/
│   └── raw/                    # Auto-downloaded dataset
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Fetching, memory optimization, preprocessing
│   ├── train_model.py          # Model training & evaluation pipeline
│   └── shap_explainer.py       # SHAP explainability utilities
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py # Tests for data pipeline
│   └── test_model.py           # Tests for model predictions
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Reproducibility

```bash
# 1. Clone the repository
git clone <repo-url>
cd obesity-risk-gr27

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the models (fetches data automatically)
python src/train_model.py

# 5. Launch the web dashboard
streamlit run app/app.py
```

---

## 💻 Usage

### Train Models

```bash
python src/train_model.py
```

This will:
- Download the dataset (if not cached)
- Optimize memory
- Train 4 classifiers
- Save the best model and evaluation metrics to `data/`

### Launch Dashboard

```bash
streamlit run app/app.py
```

### Run Tests

```bash
pytest tests/ -v
```

---

## 🔍 Exploratory Data Analysis

See the full analysis in [`notebooks/eda.ipynb`](notebooks/eda.ipynb). Key findings:

| Aspect | Finding |
|---|---|
| **Missing values** | None |
| **Outliers** | Mild and plausible — no removal needed |
| **Class balance** | Approximately balanced (~270–350 per class) |
| **Top correlated feature** | *Weight* has the strongest correlation with obesity level |
| **Memory optimization** | ~40–50 % reduction via float64 → float32 / int64 → int32 |

---

## 🤖 Models & Results

Four gradient-boosting and ensemble models were trained and compared:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.9574 | 0.9609 | 0.9574 | 0.9580 | 0.9972 |
| XGBoost | 0.9574 | 0.9597 | 0.9574 | 0.9578 | 0.9967 |
| **LightGBM** ✅ | **0.9693** | **0.9711** | **0.9693** | **0.9696** | **0.9988** |
| CatBoost | 0.9645 | 0.9666 | 0.9645 | 0.9649 | 0.9984 |

### Model Selection Justification

The best model is selected based on the highest **weighted F1-Score**, which balances precision and recall across all seven classes. All models use **class-weight balancing** to handle any residual class imbalance.

---

## 🔬 SHAP Explainability

We use **SHAP (SHapley Additive exPlanations)** with `TreeExplainer` to interpret model predictions:

- **Summary Plot** — global feature importance across all classes.
- **Feature Importance Bar Plot** — mean |SHAP| per feature.
- **Waterfall Plot** — per-patient explanation shown in the Streamlit dashboard.

SHAP explanations are integrated directly into the web interface so that physicians can understand **why** a specific obesity level was predicted.

---

## 🌐 Web Application

The Streamlit dashboard (`app/app.py`) provides:

- 🏥 **ECC-branded header** with the École Centrale Casablanca logo
- 📋 **Patient input form** (sidebar) — age, height, weight, eating habits, physical activity, etc.
- 📊 **Prediction results** — predicted obesity level, confidence score, BMI
- 📈 **Probability chart** — horizontal bar chart of class probabilities
- 🔬 **SHAP waterfall** — per-prediction explanation

**Design:** Modern medical interface with a white / blue colour palette, minimalistic layout, and Plotly interactive charts.

---

## ✅ Automated Testing & CI

### Tests

| Test File | Tests |
|---|---|
| `tests/test_data_processing.py` | `test_returns_dataframe`, `test_expected_shape`, `test_target_column_present`, `test_no_missing_values`, `test_reduces_memory`, `test_float32_conversion`, `test_int32_conversion`, `test_preserves_values` |
| `tests/test_model.py` | `test_predict_returns_single_value`, `test_predict_proba_shape`, `test_probabilities_sum_to_one`, `test_prediction_in_valid_range`, `test_model_comparison_csv_exists` |

### GitHub Actions

The CI workflow (`.github/workflows/ci.yml`) runs on every push / PR to `main`:
1. Install dependencies
2. Run data-processing tests
3. Train the model
4. Run model-prediction tests

---

## 💡 Prompt Engineering Documentation

### Prompts Used

| # | Prompt (Summarized) | Purpose |
|---|---|---|
| 1 | *"Build a complete ML pipeline for obesity estimation with 4 models, SHAP, Streamlit dashboard, and automated tests following a specific project structure"* | Full project generation |
| 2 | *Follow-up refinements for UI design, SHAP integration, and testing* | Iterative improvement |

### Analysis of Prompt Effectiveness

- **Comprehensive initial prompts** that specify structure, metrics, and design requirements produce higher-quality first drafts.
- **Modular instructions** (separate sections for EDA, ML, SHAP, UI) lead to cleaner code organization.
- **Explicit examples** (e.g. `fetch_dataset() → download → store → load`) reduce ambiguity.

### Possible Improvements

- Provide sample input/output for each function to reduce iteration.
- Specify exact hyperparameter ranges to avoid default choices.
- Include wireframes or mockups for the Streamlit UI.

---

## ❓ Required Questions

### Was the dataset balanced?
**Yes, approximately.** The original data collection covered 23 % real responses, then 77 % was augmented with SMOTE. The resulting 7 classes have between ~270 and ~350 samples each, making the dataset roughly balanced (ratio ≈ 1.3:1).

### How was class imbalance handled?
Despite the near-balance, we applied **`class_weight='balanced'`** (or its equivalent) in all classifiers as a precautionary measure. This ensures that minority classes receive proportionally higher importance during training.

### Which model performed best?
**LightGBM** achieved the highest scores across all metrics: F1-Score = **0.9696**, Accuracy = **0.9693**, ROC-AUC = **0.9988**. It was automatically selected and saved to `data/best_model.joblib`.

### Which features influenced predictions the most (SHAP)?
According to SHAP analysis, the top predictive features are:
1. **Weight** — the single strongest predictor
2. **Height**
3. **Age**
4. **Family history of overweight**
5. **FCVC** (vegetable consumption frequency)

### What insights were obtained from prompt engineering?
- Detailed, structured prompts with explicit folder layouts and function signatures dramatically reduce the need for iteration.
- Separating concerns (data, training, explainability, UI) in the prompt leads to modular, maintainable code.
- Specifying evaluation metrics and design aesthetics upfront ensures the output meets academic standards.

---

## 📜 License

This project uses data licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**Citation:**
```
Estimation of Obesity Levels Based On Eating Habits and Physical Condition [Dataset]. (2019).
UCI Machine Learning Repository. https://doi.org/10.24432/C5H31Z.