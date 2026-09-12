# Explainable Student Performance Prediction

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![XAI](https://img.shields.io/badge/Explainability-SHAP-success.svg)](https://shap.readthedocs.io/)
[![Database](https://img.shields.io/badge/Database-SQLite-003B57.svg)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **TY B.Sc. Data Science — Final Year Project**  
> An end-to-end Machine Learning and Explainable AI (XAI) system to predict student academic performance categories (`Low`, `Medium`, `High`) and explain individual and global predictions using SHAP, complete with an interactive Streamlit dashboard and local SQLite ledger.

---

## 📋 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [Project Objectives](#-project-objectives)
3. [Key Features](#-key-features)
4. [System Workflow](#-system-workflow)
5. [Anti-Data-Leakage Architecture](#-anti-data-leakage-architecture)
6. [Technologies Used](#-technologies-used)
7. [Dataset Description](#-dataset-description)
8. [Data Cleaning Pipeline](#-data-cleaning-pipeline)
9. [Machine Learning Methodology](#-machine-learning-methodology)
10. [Model Evaluation & Comparison](#-model-evaluation--comparison)
11. [Explainable AI (SHAP & Insights)](#-explainable-ai-shap--insights)
12. [Optional Local LLM Layer](#-optional-local-llm-layer)
13. [Streamlit Dashboard Guide](#-streamlit-dashboard-guide)
14. [SQLite Database Schema](#-sqlite-database-schema)
15. [Project Directory Structure](#-project-directory-structure)
16. [Installation & Setup](#-installation--setup)
17. [How to Run the Application](#-how-to-run-the-application)
18. [Automated Testing](#-automated-testing)
19. [Hardware & Software Requirements](#-hardware--software-requirements)
20. [Project Limitations & Future Scope](#-project-limitations--future-scope)

---

## 🎯 Problem Statement
In higher education, predicting student academic outcomes typically relies on historical examination marks and late manual interventions. While Machine Learning classifiers can predict student performance, most complex models function as "black boxes"—failing to explain *why* a student is flagged as high-risk or on track for distinction. This lack of interpretability prevents educators and counselors from taking targeted, timely, and trusted pedagogical actions.

---

## 🚀 Project Objectives
- Build an automated data cleaning and quality audit pipeline for real-world raw student records.
- Enforce strict anti-data-leakage separation between causal predictors and final outcomes.
- Train and evaluate primary (**Random Forest**), baseline (**Logistic Regression**), and optional (**XGBoost**) classifiers.
- Implement **SHAP (SHapley Additive exPlanations)** for global feature importance and individual student explanation breakdowns (positive vs negative drivers).
- Deploy an intuitive, multi-page **Streamlit dashboard** providing data inspection, EDA charts, live student predictions, XAI visualizations, and an SQLite historical audit ledger.
- Provide zero-cost local execution without cloud dependencies or paid APIs.

---

## 🌟 Key Features
- **Raw-to-Clean Data Pipeline**: Handles missing values, duplicates, out-of-range records (e.g., attendance > 100%), and categorical casing variations.
- **Exploratory Data Analysis (EDA)**: Interactive Plotly visual charts analyzing attendance, study habits, assignments, and participation.
- **Target Leakage Proofing**: Quarantines `Final_Score` and `Performance_Category` during feature engineering.
- **Multi-Model Benchmark**: Side-by-side comparison of Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
- **Explainability (SHAP)**: Waterfall and contribution charts separating factors that elevate or diminish a student's standing.
- **Dual-Tier Verbalizer**: Uses local `Qwen2.5-3B-Instruct` via Ollama when active, with automatic fallback to a deterministic Python rule-based template.
- **Local SQLite Audit Log**: Saves predictions, confidence scores, and XAI summaries for longitudinal tracking.

---

## 🔄 System Workflow
```
┌──────────────────────────┐
│ Raw CSV Student Data     │ (Missing values, duplicates, outliers)
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Automated Data Cleaning  │ (Deduplication, imputation, range validation)
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Strict Feature Selection │ (Quarantine Final_Score & Performance_Category)
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Train/Test Split (80/20) │ (Stratified on target classes)
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Model Training & Eval    │ ──▶ Primary: Random Forest Classifier
│                          │ ──▶ Baseline: Logistic Regression
│                          │ ──▶ Optional: XGBoost Classifier
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ SHAP Explainability      │ ──▶ Global Importance (Mean |SHAP|)
│                          │ ──▶ Local Student Factors (+ / - drivers)
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Streamlit UI Dashboard   │ ──▶ Predictions, Charts, XAI, SQLite History
└──────────────────────────┘
```

---

## 🛡️ Anti-Data-Leakage Architecture
A common flaw in academic student performance modeling is **Target Leakage**—using end-of-term metrics (e.g. final examination score or final letter grade) to predict the outcome itself.
- **Forbidden Inputs**: `Final_Score` and `Performance_Category` are strictly quarantined from the feature matrix $X$.
- **Allowed Predictors**: Only pre-assessment indicators available prior to final examination:
  1. `Attendance_Percentage`
  2. `Study_Hours` (weekly self-study)
  3. `Previous_Marks` (prior semester benchmark)
  4. `Assignment_Score` (coursework average)
  5. `Internal_Assessment_Marks` (midterm exams out of 50)
  6. `Completed_Assignments` (count out of 10)
  7. `Academic_Participation` (`Low`, `Medium`, `High`)
- **Pipeline Guarantee**: The codebase includes `verify_no_data_leakage()`, which asserts and raises exceptions if target fields ever enter model input vectors.

---

## 💻 Technologies Used
| Technology | Role | Licensing |
|---|---|---|
| **Python 3.11** | Core Programming Language | Open Source |
| **Pandas & NumPy** | Data Wrangling & Numerical Computations | BSD |
| **Scikit-learn** | Machine Learning & Preprocessing | BSD-3-Clause |
| **SHAP** | Explainable AI (Shapley Values) | MIT |
| **Streamlit** | Interactive Web Dashboard | Apache 2.0 |
| **Plotly / Seaborn** | Interactive & Statistical Visualizations | MIT / BSD |
| **SQLite** | Local Database Storage | Public Domain |
| **Joblib** | Model Serialization | BSD-3-Clause |
| **Ollama (Optional)** | Local LLM Runner (Qwen2.5-3B-Instruct) | MIT |

---

## 📊 Dataset Description
The dataset models a realistic university cohort with natural noise, human variation, and non-linear interactions:
- **Clean Dataset** (`data/processed/cleaned_student_data.csv`): 1,000 verified student records.
- **Raw Unclean Dataset** (`data/raw/unclean_student_data.csv`): 1,200 records exhibiting real-world anomalies.

### Feature Specification:
| Field Name | Type | Range | Description |
|---|---|---|---|
| `Student_ID` | String | STU1001-STU2000 | Anonymous student identifier |
| `Attendance_Percentage` | Float | 35.0% - 100.0% | Lecture and lab attendance rate |
| `Study_Hours` | Float | 1.0 - 35.0 hrs | Self-study hours per week |
| `Previous_Marks` | Float | 30.0% - 98.0% | Previous semester overall percentage |
| `Assignment_Score` | Float | 25.0% - 100.0% | Continuous assessment average |
| `Internal_Assessment_Marks`| Float | 10.0 - 50.0 | Mid-term internal exam marks |
| `Completed_Assignments` | Integer | 0 - 10 | Completed homework/project count |
| `Academic_Participation` | String | Low, Medium, High | Classroom discussion & lab engagement |
| `Final_Score` (Target) | Float | 30.0 - 99.0 | End-semester score (Regression target) |
| `Performance_Category` | String | Low, Medium, High | Classification target |

### Target Categories:
- **Low Performance**: Final Score < 50% (High Academic Risk)
- **Medium Performance**: 50% <= Final Score < 75% (Satisfactory Standing)
- **High Performance**: Final Score >= 75% (Strong Standing / Distinction)

---

## 🧹 Data Cleaning Pipeline
The module `src/data_cleaning.py` automatically executes:
1. **Deduplication**: Removes duplicate student records based on `Student_ID`.
2. **Numeric Type Coercion**: Strips stray characters (`%`, `$`, spaces) and casts text numbers into IEEE floating-point values.
3. **Boundary Validation**: Filters out impossible values (Attendance > 100% or < 0%, negative study hours, internal marks > 50).
4. **Categorical Normalization**: Strips spaces, applies title casing (`"HIGH"` / `"high "` -> `"High"`), and imputes unknown entries to the mode (`"Medium"`).
5. **Missing Value Imputation**: Continuous variables filled with robust medians; integer counts rounded to nearest median.

---

## 🤖 Machine Learning Methodology
- **Stratified Split**: 80% Training ($N=800$), 20% Testing ($N=200$) preserving class proportions.
- **Scaling**: `StandardScaler` fit strictly on $X_{train}$ and applied to $X_{test}$ to prevent train-test contamination.
- **Models**:
  1. **Random Forest (Primary)**: 100 estimators, max depth 8. Superior at capturing non-linear interactions among study hours and internal marks.
  2. **Logistic Regression (Baseline)**: Multinomial softmax with L2 regularization on standardized features.
  3. **XGBoost (Comparative)**: Gradient-boosted decision trees.

---

## 📈 Model Evaluation & Comparison
Typical performance achieved on the validated dataset (without artificial inflation):

| Metric | Random Forest (Primary) | Logistic Regression (Baseline) | XGBoost (Comparative) |
|---|---|---|---|
| **Accuracy** | **~85.5%** | ~81.0% | ~84.5% |
| **Macro Precision** | **0.86** | 0.81 | 0.85 |
| **Macro Recall** | **0.85** | 0.81 | 0.84 |
| **Macro F1-Score** | **0.85** | 0.81 | 0.84 |

Confusion matrices confirm reliable classification across all three tiers, correctly flagging low-performing students early without skewing towards the majority.

---

## 🔍 Explainable AI (SHAP & Insights)
The system uses cooperative game theory through **SHAP (SHapley Additive exPlanations)**:
$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} (f_x(S \cup \{i\}) - f_x(S))$$
- **Global Explanations**: Identifies overall driving features. Class Attendance, Previous Marks, and Internal Assessments consistently rank as top global influencers.
- **Individual Student Explanations**: Decomposes the model output into additive positive contributions ($\phi_i > 0$) pushing towards the category, and negative constraints ($\phi_i < 0$) pulling the score down.
- **Plain English Insights**: Converts mathematical Shapley values into human-readable sentences suitable for faculty and students.

---

## 🧠 Optional Local LLM Layer
- **Model**: `Qwen2.5-3B-Instruct` running locally via **Ollama** (`http://localhost:11434/api/generate`).
- **Strict Guardrail**: The LLM **never** makes predictions. It strictly verbalizes verified ML/SHAP results into constructive counseling paragraphs.
- **Zero-Dependency Fallback**: If Ollama is offline, the application seamlessly uses a deterministic Python template generator with zero loss in core functionality.

---

## 🖥️ Streamlit Dashboard Guide
Launch the application to access 7 specialized pages:
1. **Home / Overview**: System architecture, key metrics, and anti-leakage policies.
2. **Data Upload & Cleaning**: Upload CSVs, execute the cleaning pipeline, and view audit reports.
3. **Exploratory Data Analysis**: 5 interactive Plotly charts examining student trends.
4. **Model Evaluation & Comparison**: Side-by-side metric tables and confusion matrices.
5. **Student Prediction**: Live manual inputs or sample selection with risk badges and confidence meters.
6. **SHAP Explainability**: Global feature importance and individual student waterfall charts.
7. **Prediction History (SQLite)**: Searchable database log with CSV export.

---

## 🗄️ SQLite Database Schema
Database located at: `database/student_performance.db`

```sql
CREATE TABLE students (
    student_id TEXT PRIMARY KEY,
    attendance_percentage REAL NOT NULL,
    study_hours REAL NOT NULL,
    previous_marks REAL NOT NULL,
    assignment_score REAL NOT NULL,
    internal_assessment_marks REAL NOT NULL,
    completed_assignments INTEGER NOT NULL,
    academic_participation TEXT NOT NULL,
    final_score REAL,
    performance_category TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    predicted_category TEXT NOT NULL,
    predicted_score REAL,
    prediction_confidence REAL,
    top_positive_factors TEXT,
    top_negative_factors TEXT,
    explanation_summary TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 📁 Project Directory Structure
```text
explainable-student-performance/
│
├── app.py                         # Interactive Streamlit Multi-page Application
│
├── data/
│   ├── raw/
│   │   └── unclean_student_data.csv    # 1,200 records with realistic dirty anomalies
│   └── processed/
│       └── cleaned_student_data.csv    # 1,000 clean, validated student records
│
├── database/
│   ├── schema.sql                 # SQLite DDL schema
│   └── student_performance.db     # SQLite DB (auto-initialized)
│
├── models/
│   ├── random_forest_model.pkl    # Serialized Random Forest Classifier
│   ├── logistic_regression_model.pkl # Baseline Logistic Regression
│   ├── xgboost_model.pkl          # Optional XGBoost Classifier
│   ├── scaler.pkl                 # StandardScaler object
│   ├── feature_metadata.json      # Preprocessing metadata
│   └── evaluation_metrics.json    # Accuracy, Precision, Recall, F1, Confusion Matrices
│
├── notebooks/
│   └── analysis.ipynb             # Complete analytical walkthrough & EDA notebook
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py          # Synthetic realistic data generator (raw & clean)
│   ├── data_cleaning.py           # Automated data validation & cleaning pipeline
│   ├── preprocessing.py           # Train/test split, encoding, anti-leakage checks
│   ├── train_models.py            # Model training & comparative evaluation
│   ├── prediction.py              # Prediction service with risk badges
│   ├── explainability.py          # SHAP TreeExplainer & local factor decomposition
│   ├── database.py                # SQLite database management functions
│   ├── eda.py                     # Plotly EDA visualization generator
│   └── llm_explanation.py         # Ollama Qwen2.5-3B verbalizer + template fallback
│
├── tests/
│   ├── __init__.py
│   ├── test_cleaning.py           # Verification of cleaning edge cases
│   ├── test_prediction.py         # Verification of inference without leakage
│   ├── test_explainability.py     # Verification of SHAP values & explanations
│   └── test_database.py           # Verification of SQLite table creation & storage
│
├── requirements.txt               # Free & open-source Python dependencies
├── README.md                      # Complete project documentation
└── .gitignore                     # Python / data science gitignore
```

---

## ⚙️ Installation & Setup

### 1. Clone or Open Project
Navigate to the project root directory:
```bash
cd explainable-student-performance
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Application

### Step 1: Generate Datasets & Train Models
If you want to re-generate datasets and re-train models from scratch:
```bash
# 1. Generate clean and raw datasets
python src/data_generator.py

# 2. Execute automated cleaning pipeline
python src/data_cleaning.py

# 3. Train models and generate evaluation metrics
python src/train_models.py
```

### Step 2: Launch Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

### Step 3: (Optional) Local Ollama LLM Setup
If you wish to test the optional local LLM verbalizer:
1. Download Ollama from [ollama.ai](https://ollama.ai).
2. Pull and run Qwen2.5-3B:
   ```bash
   ollama run qwen2.5:3b-instruct
   ```
3. Keep Ollama running in the background. The application automatically detects `http://localhost:11434`. If absent, the built-in template verbalizer takes over seamlessly.

---

## 🧪 Automated Testing
Run the comprehensive test suite using `pytest`:
```bash
pytest tests/ -v
```
All tests verify:
- Data cleaning edge cases (duplicates, missing values, range bounds).
- Strict anti-data leakage enforcement.
- SHAP calculation and factor alignment.
- SQLite initialization, insertion, and querying.

---

## 💻 Hardware & Software Requirements
- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: Minimum 8 GB (16 GB recommended if running Ollama concurrently)
- **CPU**: Modern multi-core processor (Intel i5/Ryzen 5 or higher)
- **GPU**: Not required (all tabular models & SHAP run efficiently on CPU)
- **Storage**: ~500 MB free space

---

## 🔮 Project Limitations & Future Scope
### Limitations:
- Cross-sectional cohort data (does not currently track weekly time-series attendance fluctuations).
- Academic indicators focus on quantitative assessments without psychological sentiment data.

### Future Scope:
- Integration with institutional Learning Management Systems (Canvas, Moodle).
- Personalized remediation planner prescribing targeted topic revisions.
- Longitudinal semester-by-semester trajectory tracking.
