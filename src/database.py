"""
Database management module for Explainable Student Performance Prediction.
Handles SQLite initialization, student record storage, prediction logging, and metric tracking.
"""

import os
import sqlite3
from typing import Optional, List, Dict, Any
import pandas as pd

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "database",
    "student_performance.db"
)

SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "database",
    "schema.sql"
)


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Creates and returns a SQLite connection with Row factory."""
    path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path: Optional[str] = None) -> None:
    """Initializes the SQLite database tables using the schema file."""
    path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(path)
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()


def insert_students_batch(df: pd.DataFrame, db_path: Optional[str] = None) -> int:
    """
    Inserts or updates a batch of student records into the database.
    Returns the count of records inserted/updated.
    """
    init_database(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    query = """
        INSERT OR REPLACE INTO students (
            student_id, attendance_percentage, study_hours, previous_marks,
            assignment_score, internal_assessment_marks, completed_assignments,
            academic_participation, final_score, performance_category
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    records = []
    for _, row in df.iterrows():
        records.append((
            str(row.get("Student_ID", "")),
            float(row.get("Attendance_Percentage", 0.0)),
            float(row.get("Study_Hours", 0.0)),
            float(row.get("Previous_Marks", 0.0)),
            float(row.get("Assignment_Score", 0.0)),
            float(row.get("Internal_Assessment_Marks", 0.0)),
            int(row.get("Completed_Assignments", 0)),
            str(row.get("Academic_Participation", "Medium")),
            float(row.get("Final_Score")) if pd.notna(row.get("Final_Score")) else None,
            str(row.get("Performance_Category")) if pd.notna(row.get("Performance_Category")) else None
        ))
        
    cursor.executemany(query, records)
    conn.commit()
    count = len(records)
    conn.close()
    return count


def save_prediction(
    student_id: str,
    model_name: str,
    predicted_category: str,
    predicted_score: Optional[float] = None,
    prediction_confidence: Optional[float] = None,
    top_positive_factors: Optional[str] = "",
    top_negative_factors: Optional[str] = "",
    explanation_summary: Optional[str] = "",
    db_path: Optional[str] = None
) -> int:
    """
    Saves an individual student prediction and its XAI factors into the predictions table.
    Returns the inserted row ID.
    """
    init_database(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    query = """
        INSERT INTO predictions (
            student_id, model_name, predicted_category, predicted_score,
            prediction_confidence, top_positive_factors, top_negative_factors,
            explanation_summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.execute(query, (
        student_id,
        model_name,
        predicted_category,
        predicted_score,
        prediction_confidence,
        top_positive_factors,
        top_negative_factors,
        explanation_summary
    ))
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_predictions_history(limit: int = 200, db_path: Optional[str] = None) -> pd.DataFrame:
    """Retrieves stored predictions from the database as a Pandas DataFrame."""
    init_database(db_path)
    conn = get_db_connection(db_path)
    query = f"SELECT * FROM predictions ORDER BY timestamp DESC LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def save_model_evaluation(
    model_name: str,
    accuracy: float,
    precision_score: float,
    recall_score: float,
    f1_score: float,
    db_path: Optional[str] = None
) -> None:
    """Stores evaluation metrics for a trained machine learning model."""
    init_database(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    query = """
        INSERT INTO model_evaluations (
            model_name, accuracy, precision_score, recall_score, f1_score
        ) VALUES (?, ?, ?, ?, ?)
    """
    cursor.execute(query, (model_name, accuracy, precision_score, recall_score, f1_score))
    conn.commit()
    conn.close()


def get_model_evaluations(db_path: Optional[str] = None) -> pd.DataFrame:
    """Retrieves all model evaluations stored in the database."""
    init_database(db_path)
    conn = get_db_connection(db_path)
    query = "SELECT * FROM model_evaluations ORDER BY evaluation_date DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
