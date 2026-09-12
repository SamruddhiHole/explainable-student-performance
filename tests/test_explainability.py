"""
Automated unit tests for SHAP explainability and natural language verbalizer.
"""

import pytest
import pandas as pd
from src.explainability import ShapExplainerService, FEATURE_FRIENDLY_NAMES
from src.llm_explanation import generate_template_explanation


def test_friendly_feature_names_mapping():
    """Verifies that all standard features have teacher-friendly descriptive names."""
    from src.preprocessing import FEATURE_COLUMNS
    for feat in FEATURE_COLUMNS:
        assert feat in FEATURE_FRIENDLY_NAMES


def test_individual_shap_explanation():
    """Tests SHAP individual student explanation and positive/negative categorization."""
    explainer = ShapExplainerService()
    if explainer.rf_model is None or explainer.explainer is None:
        pytest.skip("Models or SHAP explainer not trained/ready on disk.")

    student = {
        "Attendance_Percentage": 92.0,
        "Study_Hours": 20.0,
        "Previous_Marks": 85.0,
        "Assignment_Score": 90.0,
        "Internal_Assessment_Marks": 44.0,
        "Completed_Assignments": 10,
        "Academic_Participation": "High"
    }
    result = explainer.explain_individual_student(student, predicted_class_idx=2)
    assert result["predicted_category"] == "High"
    assert len(result["factors"]) == 7
    assert isinstance(result["readable_positive"], list)
    assert isinstance(result["readable_negative"], list)


def test_template_verbalizer():
    """Verifies that the deterministic Python verbalizer produces consistent, factual text."""
    dummy_pred = {
        "predicted_category": "High",
        "confidence": 88.5,
        "predicted_score": 84.2,
        "risk_info": {"level": "Low Risk / Strong Standing"}
    }
    dummy_shap = {
        "positive_factors": [{"feature_name": "Class Attendance (%)", "actual_value": "92.0%", "shap_value": 0.25}],
        "negative_factors": []
    }
    text = generate_template_explanation(dummy_pred, dummy_shap)
    assert "High Performance" in text
    assert "88.5%" in text
    assert "Class Attendance (%)" in text
