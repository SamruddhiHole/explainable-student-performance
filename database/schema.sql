-- Schema for Explainable Student Performance Prediction System
-- Local SQLite Database: database/student_performance.db

CREATE TABLE IF NOT EXISTS students (
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

CREATE TABLE IF NOT EXISTS predictions (
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

CREATE TABLE IF NOT EXISTS model_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    accuracy REAL NOT NULL,
    precision_score REAL NOT NULL,
    recall_score REAL NOT NULL,
    f1_score REAL NOT NULL,
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
