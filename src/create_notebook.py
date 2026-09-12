"""
Generates the comprehensive analytical Jupyter Notebook 'notebooks/analysis.ipynb'.
Reflects the exact analytical methodology, EDA, training, and SHAP explainability
used throughout the main application.
"""

import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_PATH = os.path.join(PROJECT_ROOT, "notebooks", "analysis.ipynb")


def create_analysis_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Explainable Student Performance Prediction\n",
                "## Final Year Project — TY B.Sc. Data Science\n",
                "\n",
                "**Authors**: Data Science Student Project Team  \n",
                "**Primary Model**: Random Forest Classifier  \n",
                "**Baseline Model**: Logistic Regression  \n",
                "**Explainable AI**: SHAP (SHapley Additive exPlanations)  \n",
                "**Target**: Performance Category (`Low`, `Medium`, `High`)\n",
                "\n",
                "---\n",
                "\n",
                "### 🎯 Project Objectives\n",
                "1. Clean, validate, and preprocess academic records from raw CSV.\n",
                "2. Conduct Exploratory Data Analysis (EDA) to discover key academic correlations.\n",
                "3. Enforce **strict anti-data-leakage protocols** (quarantining `Final_Score` and `Performance_Category`).\n",
                "4. Train and benchmark multiple Machine Learning models (Random Forest, Logistic Regression, XGBoost).\n",
                "5. Calculate multi-class evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).\n",
                "6. Implement SHAP for global feature importance and individual student explanations."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Environment Setup and Library Imports"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.ensemble import RandomForestClassifier\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report\n",
                "import shap\n",
                "\n",
                "# Add project root to sys.path\n",
                "project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
                "if project_root not in sys.path:\n",
                "    sys.path.insert(0, project_root)\n",
                "\n",
                "# Set visual styles\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "sns.set_palette('muted')\n",
                "print(\"Libraries loaded successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Dataset Loading & Inspection\n",
                "We begin by inspecting the raw dataset (`unclean_student_data.csv`), observing data anomalies such as duplicates, missing values, and formatting issues."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "raw_path = os.path.join(project_root, 'data', 'raw', 'unclean_student_data.csv')\n",
                "df_raw = pd.read_csv(raw_path)\n",
                "print(f\"Raw dataset shape: {df_raw.shape}\")\n",
                "print(\"\\nMissing values count:\")\n",
                "print(df_raw.isnull().sum())\n",
                "print(\"\\nDuplicate rows:\", df_raw.duplicated(subset=['Student_ID']).sum())\n",
                "df_raw.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Automated Data Cleaning Pipeline\n",
                "We invoke the modular cleaning pipeline (`src.data_cleaning`) to remove duplicate records, impute missing values, filter out-of-range rows, and standardize categorical values."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.data_cleaning import clean_raw_data\n",
                "\n",
                "df_clean, audit_report = clean_raw_data(df_raw)\n",
                "print(f\"Cleaned dataset shape: {df_clean.shape}\")\n",
                "print(f\"Duplicates removed: {audit_report['duplicates_removed']}\")\n",
                "print(f\"Casing inconsistencies fixed: {audit_report['casing_inconsistencies_fixed']}\")\n",
                "df_clean.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Exploratory Data Analysis (EDA)\n",
                "Examining patterns between student habits, attendance, and their overall performance."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
                "\n",
                "# 1. Performance Category Distribution\n",
                "sns.countplot(data=df_clean, x='Performance_Category', order=['Low', 'Medium', 'High'], ax=axes[0, 0], palette=['#ef4444', '#eab308', '#22c55e'])\n",
                "axes[0, 0].set_title('Performance Category Distribution')\n",
                "\n",
                "# 2. Attendance vs Final Score\n",
                "sns.scatterplot(data=df_clean, x='Attendance_Percentage', y='Final_Score', hue='Performance_Category', hue_order=['Low', 'Medium', 'High'], palette=['#ef4444', '#eab308', '#22c55e'], alpha=0.7, ax=axes[0, 1])\n",
                "axes[0, 1].set_title('Attendance Percentage vs Final Score')\n",
                "\n",
                "# 3. Study Hours vs Final Score\n",
                "sns.scatterplot(data=df_clean, x='Study_Hours', y='Final_Score', hue='Performance_Category', hue_order=['Low', 'Medium', 'High'], palette=['#ef4444', '#eab308', '#22c55e'], alpha=0.7, ax=axes[1, 0])\n",
                "axes[1, 0].set_title('Weekly Study Hours vs Final Score')\n",
                "\n",
                "# 4. Academic Participation vs Final Score\n",
                "sns.boxplot(data=df_clean, x='Academic_Participation', y='Final_Score', order=['Low', 'Medium', 'High'], ax=axes[1, 1], palette=['#ef4444', '#3b82f6', '#22c55e'])\n",
                "axes[1, 1].set_title('Academic Participation Impact')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Correlation Heatmap\n",
                "Visualizing correlations among continuous academic features."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(9, 6))\n",
                "num_cols = ['Attendance_Percentage', 'Study_Hours', 'Previous_Marks', 'Assignment_Score', 'Internal_Assessment_Marks', 'Completed_Assignments', 'Final_Score']\n",
                "sns.heatmap(df_clean[num_cols].corr(), annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)\n",
                "plt.title('Academic Features Correlation Matrix')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Feature Engineering & Strict Anti-Data-Leakage Quarantine\n",
                "**Crucial Rule**: `Final_Score` and `Performance_Category` are ground truth targets and are **strictly excluded** from feature matrix $X$."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.preprocessing import prepare_dataset, verify_no_data_leakage\n",
                "\n",
                "prepared = prepare_dataset(df_clean, test_size=0.2, random_state=42)\n",
                "\n",
                "X_train = prepared['X_train']\n",
                "X_test = prepared['X_test']\n",
                "X_train_scaled = prepared['X_train_scaled']\n",
                "X_test_scaled = prepared['X_test_scaled']\n",
                "y_train = prepared['y_train']\n",
                "y_test = prepared['y_test']\n",
                "feature_names = prepared['feature_names']\n",
                "\n",
                "print(f\"Training instances: {X_train.shape[0]}\")\n",
                "print(f\"Testing instances: {X_test.shape[0]}\")\n",
                "print(f\"Features ({len(feature_names)}): {feature_names}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Model Training & Evaluation\n",
                "We train and benchmark:\n",
                "1. **Random Forest Classifier** (Primary)\n",
                "2. **Logistic Regression** (Baseline)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Random Forest Classifier\n",
                "rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=4, random_state=42)\n",
                "rf.fit(X_train, y_train)\n",
                "rf_preds = rf.predict(X_test)\n",
                "\n",
                "# 2. Logistic Regression (uses scaled data)\n",
                "lr = LogisticRegression(max_iter=1000, random_state=42)\n",
                "lr.fit(X_train_scaled, y_train)\n",
                "lr_preds = lr.predict(X_test_scaled)\n",
                "\n",
                "models_eval = {\n",
                "    \"Random Forest\": (y_test, rf_preds),\n",
                "    \"Logistic Regression\": (y_test, lr_preds)\n",
                "}\n",
                "\n",
                "results = []\n",
                "for m_name, (y_true, y_pred) in models_eval.items():\n",
                "    results.append({\n",
                "        \"Model\": m_name,\n",
                "        \"Accuracy\": f\"{accuracy_score(y_true, y_pred)*100:.2f}%\",\n",
                "        \"Macro Precision\": f\"{precision_score(y_true, y_pred, average='macro'):.3f}\",\n",
                "        \"Macro Recall\": f\"{recall_score(y_true, y_pred, average='macro'):.3f}\",\n",
                "        \"Macro F1\": f\"{f1_score(y_true, y_pred, average='macro'):.3f}\"\n",
                "    })\n",
                "\n",
                "pd.DataFrame(results)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Confusion Matrices"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "labels = ['Low', 'Medium', 'High']\n",
                "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n",
                "\n",
                "sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0])\n",
                "axes[0].set_title('Random Forest Confusion Matrix')\n",
                "axes[0].set_xlabel('Predicted Class')\n",
                "axes[0].set_ylabel('True Class')\n",
                "\n",
                "sns.heatmap(confusion_matrix(y_test, lr_preds), annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[1])\n",
                "axes[1].set_title('Logistic Regression Confusion Matrix')\n",
                "axes[1].set_xlabel('Predicted Class')\n",
                "axes[1].set_ylabel('True Class')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Explainable AI with SHAP (SHapley Additive exPlanations)\n",
                "SHAP applies cooperative game theory to quantify how much each student feature moved the prediction from the base value to the model's output."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "explainer = shap.TreeExplainer(rf)\n",
                "sample_X = X_test.iloc[:100]\n",
                "shap_vals = explainer.shap_values(sample_X)\n",
                "\n",
                "print(\"Global SHAP feature importance summary plot (High Performance Class):\")\n",
                "shap.summary_plot(shap_vals[2] if isinstance(shap_vals, list) else shap_vals[:, :, 2], sample_X, feature_names=feature_names, show=False)\n",
                "plt.title('SHAP Summary Plot — High Performance Class', fontsize=12)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Individual Student Explanation\n",
                "We inspect a single student and decompose the driving positive and negative factors."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.explainability import ShapExplainerService\n",
                "\n",
                "xai_service = ShapExplainerService()\n",
                "student_example = X_test.iloc[0].to_dict()\n",
                "predicted_cat_idx = rf.predict(X_test.iloc[[0]])[0]\n",
                "\n",
                "explanation = xai_service.explain_individual_student(student_example, predicted_cat_idx)\n",
                "\n",
                "print(f\"Predicted Category: {explanation['predicted_category']}\")\n",
                "print(\"\\n--- Positive Contributing Factors ---\")\n",
                "for s in explanation['readable_positive']:\n",
                "    print(s)\n",
                "print(\"\\n--- Negative Constraints ---\")\n",
                "for s in explanation['readable_negative']:\n",
                "    print(s)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Conclusion & Summary\n",
                "- Successfully cleaned raw academic data and eliminated data leakage.\n",
                "- Benchmarked primary Random Forest against baseline Logistic Regression.\n",
                "- Deployed SHAP Explainability to illuminate the black box into transparent, actionable guidance for educators and students."
            ]
        }
    ]

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    os.makedirs(os.path.dirname(NOTEBOOK_PATH), exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)

    print(f"Notebook successfully written to: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    create_analysis_notebook()
