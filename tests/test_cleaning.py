"""
Automated unit tests for data cleaning and validation pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_cleaning import clean_raw_data, REQUIRED_COLUMNS


@pytest.fixture
def sample_dirty_data():
    """Creates a synthetic dirty DataFrame for isolated testing."""
    return pd.DataFrame({
        "Student_ID": ["STU01", "STU02", "STU02", "STU03", "STU04", "STU05"],
        "Attendance_Percentage": ["85.0%", " 92.5 ", "92.5", "125.0", "-10.0", "78.0"],
        "Study_Hours": [12.0, " 15.5 ", 15.5, -4.0, 10.0, np.nan],
        "Previous_Marks": [68.0, 75.0, 75.0, 80.0, 60.0, 70.0],
        "Assignment_Score": [72.0, 80.0, 80.0, 85.0, 65.0, 75.0],
        "Internal_Assessment_Marks": [35.0, 42.0, 42.0, 95.0, 25.0, 30.0],
        "Completed_Assignments": [8, 9, 9, 15, 6, 7],
        "Academic_Participation": ["high", "HIGH", "HIGH", "low ", "INVALID", np.nan]
    })


def test_required_columns_validation():
    """Verifies that missing essential columns trigger ValueError."""
    incomplete_df = pd.DataFrame({"Student_ID": ["STU01"]})
    with pytest.raises(ValueError):
        clean_raw_data(incomplete_df)


def test_duplicate_removal(sample_dirty_data):
    """Verifies that duplicate records are identified and removed."""
    clean_df, audit = clean_raw_data(sample_dirty_data, filter_invalid_rows=False)
    assert audit["duplicates_removed"] == 1
    assert len(clean_df["Student_ID"].unique()) == len(clean_df)


def test_categorical_normalization(sample_dirty_data):
    """Verifies that Academic_Participation casing and spaces are standardized."""
    clean_df, audit = clean_raw_data(sample_dirty_data, filter_invalid_rows=False)
    for part in clean_df["Academic_Participation"]:
        assert part in ["Low", "Medium", "High"]


def test_out_of_bounds_detection(sample_dirty_data):
    """Verifies that impossible values (attendance > 100, internal marks > 50) are detected."""
    clean_df, audit = clean_raw_data(sample_dirty_data, filter_invalid_rows=True)
    # The dirty data has invalid attendance 125, -10 and invalid internal marks 95, so those rows are filtered
    assert audit["invalid_values_detected"]["Attendance_Percentage"] >= 2
    assert audit["invalid_values_detected"]["Internal_Assessment_Marks"] >= 1
    for att in clean_df["Attendance_Percentage"]:
        assert 0.0 <= att <= 100.0
    for int_m in clean_df["Internal_Assessment_Marks"]:
        assert 0.0 <= int_m <= 50.0


def test_missing_value_imputation(sample_dirty_data):
    """Verifies that NaN values in numerical and categorical columns are imputed."""
    clean_df, audit = clean_raw_data(sample_dirty_data, filter_invalid_rows=False)
    assert clean_df["Study_Hours"].isnull().sum() == 0
    assert clean_df["Academic_Participation"].isnull().sum() == 0
