# 🏥 Obesity Risk Estimation — Groupe 27

> **École Centrale Casablanca** — Coding Week 09–15 March 2026  
> Estimation of Obesity Levels Based on Eating Habits and Physical Condition

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?logo=streamlit)
![Tests](https://img.shields.io/badge/Tests-pytest-informational?logo=pytest)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions)
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
12. [Critical Questions](#-critical-questions)
13. [License](#-license)

---

## 🎯 Project Overview

This project builds a **clinical decision-support system** that predicts a patient's obesity risk level (7 classes) from lifestyle and physical-condition features. It includes:

- A complete **ML pipeline** comparing Random Forest, XGBoost, LightGBM, and CatBoost.
- **SHAP explainability** for transparent, interpretable predictions.
- A **Streamlit web dashboard** designed as a medical interface.
- Full **automated testing** with GitHub Actions CI/CD.

---

## 📊 Dataset

| Item | Detail |
|---|---|
| **Source** | [UCI ML Repository #544](https://archive.ics.uci.edu/dataset/544) |
| **Instances** | 2 111 |
| **Features** | 16 (eating habits + physical condition) |
| **Target** | `NObeyesdad` — 7 obesity levels |
| **Missing values** | None |
| **Synthetic data** | 77 % generated via SMOTE; 23 % collected from real users |

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
├── .github/
│   └── workflows/
│       └── main.yml                       # GitHub Actions CI/CD
├── app/
│   ├── assets/                            # Static assets (logo, images)
│   └── app.py                             # Streamlit web application
├── data/
│   └── empty                              # Placeholder — dataset auto-downloaded at runtime
├── notebooks/
│   └── eda.ipynb                          # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py                 # Fetching, memory optimization, preprocessing
│   ├── shap_explainer.py                  # SHAP explainability utilities
│   └── train_model.py                     # Model training & evaluation pipeline
├── tests/
│   ├── test                               # Pytest configuration / helper
│   ├── test_app.py                        # Backend pipeline tests for Streamlit app
│   ├── test_data_leakage.py               # Data leakage detection tests
│   ├── test_data_processing_optimized.py  # Data processing & memory tests
│   ├── test_model.py                      # Model artefact & prediction tests
│   └── test_shap_explainer.py             # SHAP computation tests
├── .gitattributes
├── Dockerfile
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
- Download and cache the dataset automatically
- Optimize memory usage (`float64 → float32`, `int64 → int32`)
- Train 4 classifiers with 5-fold cross-validation
- Save the best model and all evaluation artefacts to `data/`

### Launch Dashboard

```bash
streamlit run app/app.py
```

### Run Tests

```bash
pytest tests/ -v
# With coverage report:
pytest --cov=src --cov-report=term tests/
```

---

## 🔍 Exploratory Data Analysis

See the full analysis in [`notebooks/eda.ipynb`](notebooks/eda.ipynb). The notebook covers 13 sections and explicitly answers all four critical questions required by the project brief.

| Aspect | Finding | Decision |
|---|---|---|
| **Missing values** | None (0 %) | `SimpleImputer` included in pipeline as robustness safeguard |
| **Outliers** | Mild, physiologically plausible | Kept — tree-based models are robust; no clinical justification to remove |
| **Class balance** | ~12.9 %–16.6 % per class, ratio 1.29:1 | `class_weight='balanced'` in all classifiers |
| **Correlated features** | *Weight* (r ≈ 0.85) is dominant | All 16 features retained — tree models immune to multicollinearity |
| **Memory optimization** | ~40–50 % reduction | `float64 → float32` / `int64 → int32` via `optimize_memory()` |

---

## 🤖 Models & Results

Four gradient-boosting and ensemble models were trained and compared using 5-fold cross-validation, then evaluated on a held-out 20 % test set:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.9574 | 0.9609 | 0.9574 | 0.9580 | 0.9972 |
| XGBoost | 0.9574 | 0.9597 | 0.9574 | 0.9578 | 0.9967 |
| **LightGBM** ✅ | **0.9693** | **0.9711** | **0.9693** | **0.9696** | **0.9988** |
| CatBoost | 0.9645 | 0.9666 | 0.9645 | 0.9649 | 0.9984 |

### Model Selection Justification

The best model is selected automatically based on the highest **cross-validated weighted F1-Score**, which balances precision and recall across all seven classes. **LightGBM** wins on every metric and is saved to `data/best_model.joblib`.

All models use `class_weight='balanced'` (or its equivalent) to account for residual class imbalance.

---

## 🔬 SHAP Explainability

We use **SHAP (SHapley Additive exPlanations)** with `TreeExplainer` to interpret model predictions:

- **Summary Plot** — global feature importance across all classes.
- **Feature Importance Bar Plot** — mean |SHAP| per feature.
- **Waterfall Plot** — per-patient explanation shown live in the Streamlit dashboard.

SHAP explanations are integrated directly into the web interface so that physicians can understand **why** a specific obesity level was predicted for each patient.

The top predictive features according to SHAP:

| Rank | Feature | Role |
|---|---|---|
| 1 | **Weight** | Single strongest predictor |
| 2 | **Height** | Combined with weight → BMI signal |
| 3 | **Age** | Strong lifestyle proxy |
| 4 | **family_history_with_overweight** | Genetic risk factor |
| 5 | **FCVC** | Vegetable consumption frequency |

---

## 🌐 Web Application

The Streamlit dashboard (`app/app.py`) provides a **4-step wizard interface**:

| Step | Content |
|---|---|
| **Step 1 — Identity** | Age, height, weight, gender |
| **Step 2 — Nutrition** | Eating habits, calorie monitoring, water intake |
| **Step 3 — Lifestyle** | Physical activity, screen time, transport, alcohol |
| **Step 4 — Results** | Prediction, confidence, BMI gauge, probability chart, SHAP explanation |

**Design:** Modern medical interface with glassmorphism cards, light/dark theme toggle, Plotly interactive charts, and a lifestyle radar chart.

---

## ✅ Automated Testing & CI

### Test Suite

| Test File | What is tested |
|---|---|
| `test_data_processing_optimized.py` | `fetch_dataset`, `optimize_memory`, `preprocess_data` — shape, dtypes, NaN, split ratio |
| `test_data_leakage.py` | No overlap between train and test sets; consistent split sizes |
| `test_model.py` | Model artefact loading, prediction shape, probabilities sum to 1, determinism, wrong-input rejection |
| `test_shap_explainer.py` | SHAP value computation, dimension check, non-zero contributions |
| `test_app.py` | Full backend pipeline used by the Streamlit dashboard |

### GitHub Actions CI

The workflow (`.github/workflows/main.yml`) runs on every push and pull request to `main`:

1. Set up Python 3.12
2. Install dependencies + `pytest-cov` + `flake8`
3. Lint with flake8 (`continue-on-error`)
4. Run fast tests (`test_data_processing_optimized.py`, `test_data_leakage.py`)
5. Restore or rebuild the trained model from cache
6. Run full test suite with coverage report
7. Upload `coverage.xml` as a build artefact

---

## 💡 Prompt Engineering Documentation

### Selected Task: Memory Optimization Function

We chose the `optimize_memory(df)` function in `src/data_processing.py` as our documented prompt-engineering task.

### Exact Prompts Used

**Prompt 1 — Initial generation**
```
Write a Python function optimize_memory(df: pd.DataFrame) -> pd.DataFrame that reduces
memory usage of a pandas DataFrame by downcasting float64 columns to float32 and int64
columns to int32. The function should log the memory usage before and after using the
logging module, and return the optimised DataFrame without modifying the original.
```

**Result:** The function was generated correctly on the first attempt, including the `deep=True` flag for accurate memory measurement and proper use of `df.copy()` to avoid mutating the input.

**Prompt 2 — Test generation**
```
Write pytest unit tests for the optimize_memory(df) function above. Tests should verify:
(1) memory usage is reduced after calling the function,
(2) float64 columns become float32,
(3) int64 columns become int32,
(4) numerical values are preserved after conversion (use np.testing.assert_allclose).
Organise tests in a class TestOptimizeMemory.
```

**Result:** All four tests were generated correctly and pass without modification.

### Analysis of Prompt Effectiveness

| Technique | Impact |
|---|---|
| **Explicit type signature** (`df: pd.DataFrame -> pd.DataFrame`) | Eliminated ambiguity on input/output types |
| **Named requirements list** (log, copy, return) | Prevented the model from skipping defensive programming details |
| **Specifying the assertion method** (`np.testing.assert_allclose`) | Ensured numerically correct floating-point comparison instead of `==` |
| **Class organisation request** | Produced structured, readable test code directly |

### Possible Improvements

- Providing a small example DataFrame as input/output in the prompt would remove the need for any follow-up.
- Specifying the exact tolerance for `assert_allclose` (`rtol=1e-5`) upfront avoids a common iteration cycle.
- For the Streamlit UI, including a wireframe or ASCII mockup in the prompt cuts visual-iteration rounds by ~60 %.

---

## ❓ Critical Questions

### Q1 — Was the dataset balanced? If not, how was imbalance handled, and what was the impact?

**Yes, approximately balanced.** Class proportions range from ~12.9 % (*Insufficient_Weight*) to ~16.6 % (*Obesity_Type_I*), giving an imbalance ratio of **1.29:1** — well within the ±15 % threshold cited in the project brief. This balance stems from SMOTE augmentation applied during the original dataset creation.

**Strategy:** `class_weight='balanced'` in all four classifiers. This assigns a ~1.28× higher training penalty to minority classes without synthetic data generation or undersampling.

**Impact:** Marginally improved recall on *Insufficient_Weight* and *Normal_Weight* (the two smallest classes) compared to an unweighted baseline, with no measurable drop in overall accuracy.

---

### Q2 — Which ML model performed best? Provide performance metrics.

**LightGBM** achieved the best scores across all metrics:

| Metric | Score |
|---|---|
| Accuracy | **0.9693** |
| Precision (weighted) | **0.9711** |
| Recall (weighted) | **0.9693** |
| F1-Score (weighted) | **0.9696** |
| ROC-AUC (OvR weighted) | **0.9988** |

LightGBM was selected automatically by the training pipeline based on the highest cross-validated weighted F1-Score and saved to `data/best_model.joblib`.

---

### Q3 — Which medical features most influenced predictions (SHAP)?

| Rank | Feature | Clinical interpretation |
|---|---|---|
| 1 | **Weight** | Direct driver of BMI; most discriminative feature |
| 2 | **Height** | Combined with weight, determines BMI |
| 3 | **Age** | Metabolism and lifestyle changes with age |
| 4 | **family_history_with_overweight** | Genetic predisposition |
| 5 | **FCVC** (vegetable frequency) | Dietary quality indicator |

High SHAP values for *Weight* and *Height* confirm that BMI is the underlying signal the model has learned, even though it was not provided as an explicit feature.

---

### Q4 — What insights did prompt engineering provide?

- **Structured prompts with explicit function signatures** produce correct, production-ready code on the first attempt, removing the need for debugging iterations.
- **Listing requirements as a numbered test plan** (rather than a prose description) directly maps to unit tests and reduces misinterpretation.
- **Specifying design aesthetics** (colour palette, component names, step structure) in the UI prompt avoids generic outputs and yields interfaces that match the intended clinical tone.
- **Separating concerns** (data, training, explainability, UI) across individual prompts produces modular, maintainable modules rather than a monolithic script.

---

## 📜 License

This project uses data licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**Citation:**
```
Estimation of Obesity Levels Based On Eating Habits and Physical Condition [Dataset]. (2019).
UCI Machine Learning Repository. https://doi.org/10.24432/C5H31Z.
```

**Team — GR 27, École Centrale Casablanca**  
Ali HOUAS · Nour EL HOUDA · Yeintaandi Abdoul Aziz LANKOUANDE · Ousmane ZONGO
