"""
Explainable AI (XAI) Module using SHAP (SHapley Additive exPlanations).
Provides:
  1. Global Feature Importance (mean absolute SHAP across samples).
  2. Individual Student Explanation (local SHAP breakdown for the predicted class).
  3. Positive and negative driving factors identification.
  4. Human-interpretable academic insight strings.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import FEATURE_COLUMNS, REVERSE_CATEGORY_MAPPING, PARTICIPATION_MAPPING, encode_features
from src.train_models import load_model_artifacts

FEATURE_FRIENDLY_NAMES = {
    "Attendance_Percentage": "Class Attendance (%)",
    "Study_Hours": "Weekly Study Hours",
    "Previous_Marks": "Previous Semester Marks",
    "Assignment_Score": "Assignment Performance",
    "Internal_Assessment_Marks": "Internal Assessment Marks",
    "Completed_Assignments": "Assignments Completed (Count)",
    "Academic_Participation": "Academic Participation"
}


class ShapExplainerService:
    def __init__(self, background_df: Optional[pd.DataFrame] = None):
        self.artifacts = load_model_artifacts()
        self.rf_model = self.artifacts.get("random_forest")
        self.metadata = self.artifacts.get("metadata", {})
        self.feature_names = self.metadata.get("feature_names", FEATURE_COLUMNS)

        self.background_data = None
        self.explainer = None
        self._init_explainer(background_df)

    def _init_explainer(self, background_df: Optional[pd.DataFrame] = None):
        """Initializes SHAP TreeExplainer on Random Forest."""
        if self.rf_model is None:
            return

        if background_df is not None:
            cols = [c for c in self.feature_names if c in background_df.columns]
            bg = background_df[cols].copy()
            if "Academic_Participation" in bg.columns and bg["Academic_Participation"].dtype == object:
                bg["Academic_Participation"] = bg["Academic_Participation"].map(PARTICIPATION_MAPPING).fillna(1).astype(int)
            self.background_data = bg
        else:
            # Try to load test dataset saved during training
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            test_path = os.path.join(project_root, "models", "test_dataset.csv")
            if os.path.exists(test_path):
                test_df = pd.read_csv(test_path)
                cols = [c for c in self.feature_names if c in test_df.columns]
                bg = test_df[cols].copy()
                if "Academic_Participation" in bg.columns and bg["Academic_Participation"].dtype == object:
                    bg["Academic_Participation"] = bg["Academic_Participation"].map(PARTICIPATION_MAPPING).fillna(1).astype(int)
                self.background_data = bg
            else:
                self.background_data = None

        try:
            # TreeExplainer is ideal and fast for Random Forest
            self.explainer = shap.TreeExplainer(self.rf_model)
        except Exception as e:
            print(f"Error initializing TreeExplainer: {e}")
            self.explainer = None

    def get_global_feature_importance(self, df_sample: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculates global SHAP importance (mean absolute SHAP across all classes and instances).
        Returns a DataFrame sorted from most important to least important feature.
        """
        if self.explainer is None:
            raise RuntimeError("SHAP explainer is not initialized. Ensure models are trained.")

        data_to_explain = df_sample if df_sample is not None else self.background_data
        if data_to_explain is None:
            raise ValueError("No background or sample data provided for global SHAP calculation.")

        # Ensure correct column order
        X = data_to_explain[self.feature_names].copy()
        if "Academic_Participation" in X.columns and X["Academic_Participation"].dtype == object:
            X["Academic_Participation"] = X["Academic_Participation"].map(PARTICIPATION_MAPPING).fillna(1).astype(int)

        shap_values = self.explainer.shap_values(X)

        # In shap multi-class: shap_values is a list of arrays [n_classes], each shape (n_samples, n_features)
        # or a 3D array (n_samples, n_features, n_classes)
        if isinstance(shap_values, list):
            # Average absolute values across classes and samples
            abs_per_class = [np.abs(sv).mean(axis=0) for sv in shap_values]
            mean_abs = np.mean(abs_per_class, axis=0)
        elif len(np.shape(shap_values)) == 3:
            mean_abs = np.abs(shap_values).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Feature_Name": [FEATURE_FRIENDLY_NAMES.get(f, f) for f in self.feature_names],
            "Mean_Absolute_SHAP": np.round(mean_abs, 4)
        }).sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)

        return importance_df

    def explain_individual_student(
        self,
        student_features: Dict[str, Any],
        predicted_class_idx: int
    ) -> Dict[str, Any]:
        """
        Generates local SHAP explanation for a single student.
        Breaks down contributions for the predicted class into positive and negative factors.
        """
        if self.explainer is None:
            raise RuntimeError("SHAP explainer is not initialized.")

        # Prepare 1-row DataFrame
        row_dict = {col: student_features.get(col, 0) for col in self.feature_names}
        df_row = pd.DataFrame([row_dict])
        df_row = encode_features(df_row).astype(float)

        shap_raw = self.explainer.shap_values(df_row)

        # Extract values for the target class
        if isinstance(shap_raw, list):
            class_shap = shap_raw[predicted_class_idx][0]
        elif len(np.shape(shap_raw)) == 3:
            class_shap = shap_raw[0, :, predicted_class_idx]
        else:
            class_shap = shap_raw[0]

        # Extract base value
        if hasattr(self.explainer, "expected_value"):
            exp_val = self.explainer.expected_value
            if isinstance(exp_val, (list, np.ndarray)):
                base_val = float(exp_val[predicted_class_idx])
            else:
                base_val = float(exp_val)
        else:
            base_val = 0.333

        factors = []
        positive_factors = []
        negative_factors = []

        rev_part = {0: "Low", 1: "Medium", 2: "High"}

        for f_idx, feat in enumerate(self.feature_names):
            actual_val = student_features.get(feat, df_row[feat].iloc[0])
            contrib = float(class_shap[f_idx])
            friendly = FEATURE_FRIENDLY_NAMES.get(feat, feat)

            # Format display value
            if feat == "Academic_Participation":
                if isinstance(actual_val, (int, float, np.integer)):
                    display_val = rev_part.get(int(actual_val), "Medium")
                else:
                    display_val = str(actual_val)
            elif feat in ["Attendance_Percentage", "Previous_Marks", "Assignment_Score"]:
                display_val = f"{float(actual_val):.1f}%"
            elif feat == "Study_Hours":
                display_val = f"{float(actual_val):.1f} hrs/week"
            elif feat == "Internal_Assessment_Marks":
                display_val = f"{float(actual_val):.1f} / 50"
            else:
                display_val = str(actual_val)

            factor_obj = {
                "feature": feat,
                "feature_name": friendly,
                "actual_value": display_val,
                "raw_value": actual_val,
                "shap_value": round(contrib, 4),
                "is_positive": contrib >= 0
            }
            factors.append(factor_obj)

            if contrib > 0.005:
                positive_factors.append(factor_obj)
            elif contrib < -0.005:
                negative_factors.append(factor_obj)

        # Sort by absolute SHAP magnitude
        positive_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        negative_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        target_cat = REVERSE_CATEGORY_MAPPING.get(predicted_class_idx, "Medium")

        # Generate teacher/student readable statements
        readable_pos = [
            f"[+] {f['feature_name']} ({f['actual_value']}) contributed positively (+{f['shap_value']:.2f}) toward {target_cat} performance"
            for f in positive_factors[:3]
        ]
        readable_neg = [
            f"[-] {f['feature_name']} ({f['actual_value']}) pulled prediction down ({f['shap_value']:.2f})"
            for f in negative_factors[:3]
        ]

        return {
            "predicted_category": target_cat,
            "base_value": round(base_val, 4),
            "factors": factors,
            "positive_factors": positive_factors,
            "negative_factors": negative_factors,
            "readable_positive": readable_pos,
            "readable_negative": readable_neg
        }


def plot_shap_waterfall_interactive(ind_expl: Dict[str, Any]) -> "go.Figure":
    """
    Renders an authentic, interactive SHAP Waterfall chart using Plotly.
    Shows the step-by-step path from Class Average Baseline to Final Student Probability.
    """
    import plotly.graph_objects as go

    base_val = ind_expl.get("base_value", 0.33)
    factors = ind_expl.get("factors", [])
    target_cat = ind_expl.get("predicted_category", "Medium")

    x_labels = ["Class Baseline<br>(Average Student)"]
    y_values = [base_val]
    measures = ["absolute"]
    texts = [f"{base_val:.2f}"]

    for f in factors:
        feat_name = f.get('feature_name', '')
        actual = f.get('actual_value', '')
        val = f.get('shap_value', 0.0)
        x_labels.append(f"{feat_name}<br><span style='font-size:11px;color:gray'>({actual})</span>")
        y_values.append(val)
        measures.append("relative")
        texts.append(f"{val:+.2f}")

    x_labels.append(f"Final Prediction<br><b>({target_cat})</b>")
    y_values.append(0)
    measures.append("total")
    texts.append("Final")

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=x_labels,
        textposition="outside",
        text=texts,
        y=y_values,
        connector={"line": {"color": "#9ca3af"}},
        decreasing={"marker": {"color": "#ef4444"}},
        increasing={"marker": {"color": "#10b981"}},
        totals={"marker": {"color": "#3b82f6"}}
    ))

    fig.update_layout(
        title=f"Step-by-Step AI Decision Journey toward '{target_cat}' Standing",
        showlegend=False,
        template="plotly_white",
        height=430,
        yaxis_title="Probability Score (0.0 to 1.0)",
        xaxis_tickangle=-20,
        margin=dict(l=40, r=40, t=50, b=80)
    )
    return fig


def plot_shap_driver_bars(ind_expl: Dict[str, Any]) -> "go.Figure":
    """
    Renders a clear diverging horizontal bar chart showing positive vs negative factors.
    """
    import plotly.graph_objects as go

    factors = ind_expl.get("factors", [])
    target_cat = ind_expl.get("predicted_category", "Medium")

    names = []
    values = []
    colors = []
    texts = []

    # Sort so most impactful are on top
    sorted_factors = sorted(factors, key=lambda x: x["shap_value"])

    for f in sorted_factors:
        val = f["shap_value"]
        names.append(f"{f['feature_name']} ({f['actual_value']})")
        values.append(val)
        if val >= 0:
            colors.append("#10b981")  # Green
            texts.append(f"+{val:.2f} (Helped)")
        else:
            colors.append("#ef4444")  # Red
            texts.append(f"{val:.2f} (Hurt)")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names,
        x=values,
        orientation="h",
        text=texts,
        textposition="outside",
        marker=dict(color=colors, line=dict(color="#ffffff", width=1.5))
    ))

    fig.update_layout(
        title=f"Key Academic Drivers: What Helped (+) vs Hurt (-) This Student",
        xaxis_title="Influence on Predicted Outcome (SHAP Value)",
        yaxis_title="Academic Factor & Student Value",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=60, t=50, b=40)
    )
    return fig


