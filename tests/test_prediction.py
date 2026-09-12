"""
Automated unit tests for prediction pipeline and anti-data leakage enforcement.
"""

import pytest
import pandas as pd
from src.preprocessing import verify_no_data_leakage, transform_single_student, FEATURE_COLUMNS
from src.prediction import PredictionService


def test_anti_data_leakage_detector():
    """Confirms that including target columns in feature inputs raises ValueError."""
    leakage_inputs = ["Attendance_Percentage", "Final_Score", "Study_Hours"]
    with pytest.raises(ValueError) as excinfo:
        verify_no_data_leakage(leakage_inputs)
    assert "TARGET LEAKAGE DETECTED" in str(excinfo.value)

    category_leakage = ["Previous_Marks", "Performance_Category"]
    with pytest.raises(ValueError) as excinfo2:
        verify_no_data_leakage(category_leakage)
    assert "TARGET LEAKAGE DETECTED" in str(excinfo2.value)


def test_feature_transformation():
    """Verifies that a student record maps to expected numerical feature vector."""
    student_record = {
        "Attendance_Percentage": 85.0,
        "Study_Hours": 14.0,
        "Previous_Marks": 70.0,
        "Assignment_Score": 75.0,
        "Internal_Assessment_Marks": 35.0,
        "Completed_Assignments": 8,
        "Academic_Participation": "High"
    }
    encoded_df, vec = transform_single_student(student_record, FEATURE_COLUMNS)
    assert encoded_df.shape == (1, len(FEATURE_COLUMNS))
    assert encoded_df["Academic_Participation"].iloc[0] == 2  # High -> 2


def test_prediction_service_probabilities():
    """Verifies that prediction service produces valid probabilities summing to ~1.0."""
    service = PredictionService()
    if not service.is_ready():
        pytest.skip("Models not yet trained on disk.")

    student_record = {
        "Attendance_Percentage": 88.0,
        "Study_Hours": 18.0,
        "Previous_Marks": 78.0,
        "Assignment_Score": 82.0,
        "Internal_Assessment_Marks": 40.0,
        "Completed_Assignments": 9,
        "Academic_Participation": "High"
    }
    res = service.predict_student(student_record, model_name="Random Forest")
    assert res["predicted_category"] in ["Low", "Medium", "High"]
    assert 0.0 <= res["confidence"] <= 100.0
    prob_sum = sum(res["probabilities"].values())
    assert pytest.approx(prob_sum, 0.01) == 1.0
