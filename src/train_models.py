"""
Model Training and Comparative Evaluation Module.
Trains:
  1. Primary Model: Random Forest Classifier
  2. Baseline Model: Logistic Regression Classifier
  3. Optional Model: XGBoost Classifier
Evaluates Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
Persists model binaries and performance benchmarks.
"""

import os
import sys
import json
import joblib
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.preprocessing import prepare_dataset, CATEGORY_MAPPING, REVERSE_CATEGORY_MAPPING
from src.database import save_model_evaluation, init_database
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CLEAN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_student_data.csv")


def train_and_evaluate_models(
    df: Optional[pd.DataFrame] = None,
    save_artifacts: bool = True
) -> Dict[str, Any]:
    """
    Executes training of Random Forest, Logistic Regression, and XGBoost.
    Calculates multi-class classification metrics.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    init_database()

    if df is None:
        if not os.path.exists(CLEAN_DATA_PATH):
            raise FileNotFoundError(f"Cleaned dataset not found at {CLEAN_DATA_PATH}. Run data generation/cleaning first.")
        df = pd.read_csv(CLEAN_DATA_PATH)

    # Prepare datasets with anti-leakage protection
    prepared = prepare_dataset(df, test_size=0.2, random_state=42)

    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    X_train_scaled = prepared["X_train_scaled"]
    X_test_scaled = prepared["X_test_scaled"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]
    feature_names = prepared["feature_names"]
    target_names = prepared["target_names"]
    scaler = prepared["scaler"]

    models = {}
    metrics_summary = {}

    # 1. Primary Model: Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=4,
        random_state=42
    )
    # Train on unscaled or scaled (RF works on raw features preserving native units)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)

    rf_metrics = {
        "accuracy": float(accuracy_score(y_test, rf_preds)),
        "precision_macro": float(precision_score(y_test, rf_preds, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, rf_preds, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_test, rf_preds, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, rf_preds, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_test, rf_preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, rf_preds, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, rf_preds).tolist(),
        "is_primary": True
    }
    models["Random Forest"] = rf_model
    metrics_summary["Random Forest"] = rf_metrics

    # 2. Baseline Model: Logistic Regression (requires scaled features)
    lr_model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42
    )
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)
    lr_probs = lr_model.predict_proba(X_test_scaled)

    lr_metrics = {
        "accuracy": float(accuracy_score(y_test, lr_preds)),
        "precision_macro": float(precision_score(y_test, lr_preds, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, lr_preds, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_test, lr_preds, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, lr_preds, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_test, lr_preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, lr_preds, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, lr_preds).tolist(),
        "is_primary": False
    }
    models["Logistic Regression"] = lr_model
    metrics_summary["Logistic Regression"] = lr_metrics

    # 3. Optional Stronger Model: XGBoost Classifier
    if XGBOOST_AVAILABLE:
        try:
            xgb_model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                random_state=42,
                eval_metric="mlogloss"
            )
            xgb_model.fit(X_train, y_train)
            xgb_preds = xgb_model.predict(X_test)
            xgb_probs = xgb_model.predict_proba(X_test)

            xgb_metrics = {
                "accuracy": float(accuracy_score(y_test, xgb_preds)),
                "precision_macro": float(precision_score(y_test, xgb_preds, average="macro", zero_division=0)),
                "precision_weighted": float(precision_score(y_test, xgb_preds, average="weighted", zero_division=0)),
                "recall_macro": float(recall_score(y_test, xgb_preds, average="macro", zero_division=0)),
                "recall_weighted": float(recall_score(y_test, xgb_preds, average="weighted", zero_division=0)),
                "f1_macro": float(f1_score(y_test, xgb_preds, average="macro", zero_division=0)),
                "f1_weighted": float(f1_score(y_test, xgb_preds, average="weighted", zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test, xgb_preds).tolist(),
                "is_primary": False
            }
            models["XGBoost"] = xgb_model
            metrics_summary["XGBoost"] = xgb_metrics
        except Exception as e:
            print(f"XGBoost training encountered an issue: {e}")

    # Record evaluation results into SQLite database
    for m_name, m_val in metrics_summary.items():
        save_model_evaluation(
            model_name=m_name,
            accuracy=m_val["accuracy"],
            precision_score=m_val["precision_macro"],
            recall_score=m_val["recall_macro"],
            f1_score=m_val["f1_macro"]
        )

    # Persist artifacts if requested
    if save_artifacts:
        joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest_model.pkl"))
        joblib.dump(lr_model, os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))
        if "XGBoost" in models:
            joblib.dump(models["XGBoost"], os.path.join(MODELS_DIR, "xgboost_model.pkl"))

        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

        metadata = {
            "feature_names": feature_names,
            "target_names": target_names,
            "category_mapping": CATEGORY_MAPPING,
            "reverse_category_mapping": REVERSE_CATEGORY_MAPPING,
            "train_size": len(X_train),
            "test_size": len(X_test)
        }
        with open(os.path.join(MODELS_DIR, "feature_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        with open(os.path.join(MODELS_DIR, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)

        # Also save test partition for fast dashboard evaluation / XAI exploration
        test_df = X_test.copy()
        test_df["Performance_Category"] = [REVERSE_CATEGORY_MAPPING[v] for v in y_test]
        test_df["Student_ID"] = prepared["test_student_ids"]
        test_df.to_csv(os.path.join(MODELS_DIR, "test_dataset.csv"), index=False)

    return {
        "models": models,
        "metrics": metrics_summary,
        "prepared": prepared
    }


def load_model_artifacts() -> Dict[str, Any]:
    """Loads saved models, scaler, and metadata from disk."""
    artifacts = {}
    rf_path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    lr_path = os.path.join(MODELS_DIR, "logistic_regression_model.pkl")
    xgb_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    meta_path = os.path.join(MODELS_DIR, "feature_metadata.json")
    eval_path = os.path.join(MODELS_DIR, "evaluation_metrics.json")

    if os.path.exists(rf_path):
        artifacts["random_forest"] = joblib.load(rf_path)
    if os.path.exists(lr_path):
        artifacts["logistic_regression"] = joblib.load(lr_path)
    if os.path.exists(xgb_path):
        artifacts["xgboost"] = joblib.load(xgb_path)
    if os.path.exists(scaler_path):
        artifacts["scaler"] = joblib.load(scaler_path)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            artifacts["metadata"] = json.load(f)
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            artifacts["metrics"] = json.load(f)

    return artifacts


if __name__ == "__main__":
    results = train_and_evaluate_models()
    for name, met in results["metrics"].items():
        print(f"{name}: Accuracy={met['accuracy']:.4f}, F1(macro)={met['f1_macro']:.4f}")
