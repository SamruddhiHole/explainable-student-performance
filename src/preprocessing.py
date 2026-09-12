"""
Preprocessing and Anti-Data-Leakage Module.
Prepares feature matrices, applies feature scaling & encoding fit strictly on training data,
and enforces anti-leakage isolation of target variables.
"""

import os
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Explicit target columns that MUST NEVER be present in feature set X
FORBIDDEN_LEAKAGE_COLUMNS = [
    "Final_Score",
    "Performance_Category",
    "final_score",
    "performance_category"
]

FEATURE_COLUMNS = [
    "Attendance_Percentage",
    "Study_Hours",
    "Previous_Marks",
    "Assignment_Score",
    "Internal_Assessment_Marks",
    "Completed_Assignments",
    "Academic_Participation"
]

NUMERICAL_FEATURES = [
    "Attendance_Percentage",
    "Study_Hours",
    "Previous_Marks",
    "Assignment_Score",
    "Internal_Assessment_Marks",
    "Completed_Assignments"
]

PARTICIPATION_MAPPING = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

REVERSE_PARTICIPATION_MAPPING = {v: k for k, v in PARTICIPATION_MAPPING.items()}

CATEGORY_MAPPING = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

REVERSE_CATEGORY_MAPPING = {0: "Low", 1: "Medium", 2: "High"}


def verify_no_data_leakage(feature_cols: List[str]) -> None:
    """
    Asserts that no target variables or future indicators are included
    in the predictive feature set.
    """
    for col in feature_cols:
        col_clean = col.strip().lower()
        for forbidden in FORBIDDEN_LEAKAGE_COLUMNS:
            if col_clean == forbidden.lower():
                raise ValueError(
                    f"CRITICAL TARGET LEAKAGE DETECTED! '{col}' is a ground truth target "
                    f"and cannot be used as an input feature for prediction."
                )


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features.
    Academic_Participation is ordinally mapped: Low->0, Medium->1, High->2.
    """
    df_encoded = df.copy()
    if "Academic_Participation" in df_encoded.columns:
        df_encoded["Academic_Participation"] = (
            df_encoded["Academic_Participation"]
            .astype(str)
            .str.strip()
            .map(PARTICIPATION_MAPPING)
            .fillna(1)
            .astype(int)
        )
    return df_encoded


def prepare_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Prepares train and test splits, verifies anti-leakage, and fits feature scaler.

    Returns dictionary containing:
        - X_train, X_test (raw values)
        - X_train_scaled, X_test_scaled (standardized values)
        - y_train, y_test (encoded integers 0, 1, 2)
        - y_train_labels, y_test_labels (strings 'Low', 'Medium', 'High')
        - scaler (fitted StandardScaler)
        - feature_names (list of features)
        - target_names (['Low', 'Medium', 'High'])
        - test_student_ids (list of student IDs in test set)
    """
    # 1. Feature selection: ensure only pre-performance variables are included
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    verify_no_data_leakage(available_features)

    X_raw = df[available_features].copy()
    X_encoded = encode_features(X_raw)

    # 2. Extract classification target
    if "Performance_Category" not in df.columns:
        raise ValueError("Target column 'Performance_Category' not found in dataset.")

    y_raw = df["Performance_Category"].astype(str).str.strip()
    y_encoded = y_raw.map(CATEGORY_MAPPING).fillna(1).astype(int).values

    # Track student IDs if present
    student_ids = df["Student_ID"].tolist() if "Student_ID" in df.columns else [f"STU{i}" for i in range(len(df))]

    # 3. Stratified Train / Test Split
    (
        X_train, X_test,
        y_train, y_test,
        ids_train, ids_test
    ) = train_test_split(
        X_encoded, y_encoded, student_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    # 4. Fit StandardScaler strictly on training split
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame for clean column names in SHAP and inspection
    X_train_df = pd.DataFrame(X_train, columns=available_features, index=X_train.index)
    X_test_df = pd.DataFrame(X_test, columns=available_features, index=X_test.index)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=available_features, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=available_features, index=X_test.index)

    return {
        "X_train": X_train_df,
        "X_test": X_test_df,
        "X_train_scaled": X_train_scaled_df,
        "X_test_scaled": X_test_scaled_df,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_labels": [REVERSE_CATEGORY_MAPPING[val] for val in y_train],
        "y_test_labels": [REVERSE_CATEGORY_MAPPING[val] for val in y_test],
        "scaler": scaler,
        "feature_names": available_features,
        "target_names": ["Low", "Medium", "High"],
        "train_student_ids": ids_train,
        "test_student_ids": ids_test
    }


def transform_single_student(
    student_dict: Dict[str, Any],
    feature_names: List[str],
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Transforms a single student input record into feature vectors for model inference.

    Returns:
        (df_encoded, scaled_array)
    """
    verify_no_data_leakage(list(student_dict.keys()))

    row_data = {}
    for col in feature_names:
        row_data[col] = student_dict.get(col, 0)

    single_df = pd.DataFrame([row_data])
    encoded_df = encode_features(single_df)

    if scaler is not None:
        scaled_vec = scaler.transform(encoded_df)
    else:
        scaled_vec = encoded_df.values

    return encoded_df, scaled_vec
