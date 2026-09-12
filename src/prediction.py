"""
Prediction Service Module.
Performs single-student and batch inferences using trained models.
Calculates class probabilities, risk indicators, and predicted score estimates.
"""

import os
import sys
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (
    FEATURE_COLUMNS,
    CATEGORY_MAPPING,
    REVERSE_CATEGORY_MAPPING,
    transform_single_student,
    encode_features,
    verify_no_data_leakage
)
from src.train_models import load_model_artifacts

RISK_LEVELS = {
    "Low": {"level": "High Academic Risk", "color": "#ef4444", "badge": "🔴 High Risk"},
    "Medium": {"level": "Moderate / Satisfactory", "color": "#eab308", "badge": "🟡 Moderate"},
    "High": {"level": "Low Risk / Strong Standing", "color": "#22c55e", "badge": "🟢 Low Risk"}
}


class PredictionService:
    def __init__(self):
        self.artifacts = load_model_artifacts()
        self.models = {
            "Random Forest": self.artifacts.get("random_forest"),
            "Logistic Regression": self.artifacts.get("logistic_regression"),
        }
        if "xgboost" in self.artifacts:
            self.models["XGBoost"] = self.artifacts.get("xgboost")
        self.scaler = self.artifacts.get("scaler")
        self.metadata = self.artifacts.get("metadata", {})
        self.feature_names = self.metadata.get("feature_names", FEATURE_COLUMNS)

    def is_ready(self) -> bool:
        """Returns True if primary model and scaler are loaded."""
        return self.models.get("Random Forest") is not None

    def predict_student(
        self,
        student_data: Dict[str, Any],
        model_name: str = "Random Forest"
    ) -> Dict[str, Any]:
        """
        Runs inference on a single student profile.
        Returns:
          - predicted_category: 'Low', 'Medium', or 'High'
          - probabilities: dict of class -> float probability
          - confidence: float confidence of the top predicted class
          - predicted_score: estimated continuous score
          - risk_info: risk badge, color, and description
        """
        if not self.is_ready():
            raise RuntimeError("Models are not trained or loaded yet. Run model training first.")

        selected_model = self.models.get(model_name)
        if selected_model is None:
            selected_model = self.models.get("Random Forest")
            model_name = "Random Forest"

        encoded_df, scaled_vec = transform_single_student(
            student_data, self.feature_names, self.scaler
        )

        # Logistic Regression uses scaled features; Tree models use unscaled
        if model_name == "Logistic Regression":
            input_features = scaled_vec
        else:
            input_features = encoded_df

        # Generate prediction and probabilities
        pred_idx = selected_model.predict(input_features)[0]
        pred_category = REVERSE_CATEGORY_MAPPING.get(int(pred_idx), "Medium")

        probs = {}
        if hasattr(selected_model, "predict_proba"):
            raw_probs = selected_model.predict_proba(input_features)[0]
            for idx, prob in enumerate(raw_probs):
                cat_name = REVERSE_CATEGORY_MAPPING.get(idx, f"Class {idx}")
                probs[cat_name] = round(float(prob), 4)
            confidence = float(max(raw_probs))
        else:
            confidence = 1.0
            probs[pred_category] = 1.0

        # Estimated continuous score derived from class probabilities & benchmark points:
        # Low centroid ~42, Medium centroid ~63, High centroid ~86
        est_score = (
            probs.get("Low", 0.0) * 42.0 +
            probs.get("Medium", 0.0) * 63.5 +
            probs.get("High", 0.0) * 86.5
        )
        # Combine slightly with direct linear proxy for fine granularity
        raw_att = float(student_data.get("Attendance_Percentage", 75.0))
        raw_prev = float(student_data.get("Previous_Marks", 65.0))
        raw_assign = float(student_data.get("Assignment_Score", 70.0))
        raw_int = float(student_data.get("Internal_Assessment_Marks", 30.0)) * 2.0
        heuristic_score = 0.25 * raw_prev + 0.20 * raw_att + 0.20 * raw_assign + 0.25 * raw_int + 0.10 * 45.0
        final_est_score = round(float(0.65 * est_score + 0.35 * heuristic_score), 1)

        risk_info = RISK_LEVELS.get(pred_category, RISK_LEVELS["Medium"])

        return {
            "model_name": model_name,
            "predicted_category": pred_category,
            "confidence": round(confidence * 100, 1),
            "probabilities": probs,
            "predicted_score": final_est_score,
            "risk_info": risk_info,
            "encoded_features": encoded_df.to_dict(orient="records")[0]
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        model_name: str = "Random Forest"
    ) -> pd.DataFrame:
        """Runs batch inference on a DataFrame of student records."""
        results = []
        for _, row in df.iterrows():
            res = self.predict_student(row.to_dict(), model_name=model_name)
            results.append({
                "Student_ID": row.get("Student_ID", ""),
                "Predicted_Category": res["predicted_category"],
                "Confidence_%": res["confidence"],
                "Predicted_Score": res["predicted_score"],
                "Risk_Level": res["risk_info"]["level"]
            })
        return pd.DataFrame(results)
