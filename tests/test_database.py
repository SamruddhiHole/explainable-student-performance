"""
Automated unit tests for SQLite database operations.
"""

import os
import sqlite3
import pytest
import pandas as pd
from src.database import (
    init_database,
    insert_students_batch,
    save_prediction,
    get_predictions_history,
    save_model_evaluation,
    get_model_evaluations
)


@pytest.fixture
def temp_db_path(tmp_path):
    """Creates a temporary SQLite database path for isolated test runs."""
    return str(tmp_path / "test_student_performance.db")


def test_init_database(temp_db_path):
    """Verifies table creation schema."""
    init_database(temp_db_path)
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    assert "students" in tables
    assert "predictions" in tables
    assert "model_evaluations" in tables


def test_insert_students_batch(temp_db_path):
    """Verifies batch insertion of student records."""
    init_database(temp_db_path)
    df = pd.DataFrame({
        "Student_ID": ["STU901", "STU902"],
        "Attendance_Percentage": [85.0, 78.0],
        "Study_Hours": [12.0, 10.0],
        "Previous_Marks": [70.0, 65.0],
        "Assignment_Score": [75.0, 68.0],
        "Internal_Assessment_Marks": [38.0, 32.0],
        "Completed_Assignments": [8, 7],
        "Academic_Participation": ["High", "Medium"],
        "Final_Score": [76.5, 62.0],
        "Performance_Category": ["High", "Medium"]
    })
    count = insert_students_batch(df, temp_db_path)
    assert count == 2


def test_save_and_retrieve_prediction(temp_db_path):
    """Verifies prediction persistence and retrieval from SQLite."""
    init_database(temp_db_path)
    row_id = save_prediction(
        student_id="STU999",
        model_name="Random Forest",
        predicted_category="High",
        predicted_score=85.5,
        prediction_confidence=91.2,
        top_positive_factors="Attendance",
        top_negative_factors="None",
        explanation_summary="High standing",
        db_path=temp_db_path
    )
    assert row_id > 0

    history = get_predictions_history(limit=10, db_path=temp_db_path)
    assert len(history) == 1
    assert history.iloc[0]["student_id"] == "STU999"
    assert history.iloc[0]["predicted_category"] == "High"
