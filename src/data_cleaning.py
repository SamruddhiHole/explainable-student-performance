"""
Automated Data Cleaning and Quality Pipeline.
Transforms raw unclean student records into a standardized, validated dataset.
Provides full transparency metrics: duplicates removed, missing values imputed,
and out-of-bounds values handled.
"""

import os
import re
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "unclean_student_data.csv")
DEFAULT_CLEAN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_student_data.csv")

VALID_RANGES = {
    "Attendance_Percentage": (0.0, 100.0),
    "Study_Hours": (0.0, 60.0),
    "Previous_Marks": (0.0, 100.0),
    "Assignment_Score": (0.0, 100.0),
    "Internal_Assessment_Marks": (0.0, 50.0),
    "Completed_Assignments": (0, 10)
}

REQUIRED_COLUMNS = [
    "Student_ID",
    "Attendance_Percentage",
    "Study_Hours",
    "Previous_Marks",
    "Assignment_Score",
    "Internal_Assessment_Marks",
    "Completed_Assignments",
    "Academic_Participation"
]


def clean_raw_data(
    df_raw: pd.DataFrame,
    filter_invalid_rows: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the end-to-end data cleaning pipeline.

    Args:
        df_raw: Raw DataFrame with potential dirty data anomalies.
        filter_invalid_rows: If True, completely unrecoverable/extreme invalid records
                             are dropped; otherwise clipped/imputed.

    Returns:
        (df_clean, cleaning_audit_report)
    """
    report: Dict[str, Any] = {
        "raw_shape": df_raw.shape,
        "duplicates_removed": 0,
        "missing_values_before": {},
        "invalid_values_detected": {},
        "casing_inconsistencies_fixed": 0,
        "cleaned_shape": (0, 0)
    }

    df = df_raw.copy()

    # 1. Standardize column names: strip whitespace
    df.columns = [c.strip() for c in df.columns]

    # Verify presence of required columns
    missing_req = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_req:
        raise ValueError(f"Dataset is missing required columns: {missing_req}")

    # 2. Count missing values before any processing
    report["missing_values_before"] = df[REQUIRED_COLUMNS].isnull().sum().to_dict()

    # 3. Detect and remove duplicate records
    init_count = len(df)
    # Check duplicates on Student_ID or whole row
    df = df.drop_duplicates(subset=["Student_ID"], keep="first").reset_index(drop=True)
    report["duplicates_removed"] = init_count - len(df)

    # 4. Clean and parse numeric fields stored as strings
    numeric_cols = [
        "Attendance_Percentage", "Study_Hours", "Previous_Marks",
        "Assignment_Score", "Internal_Assessment_Marks", "Completed_Assignments"
    ]
    if "Final_Score" in df.columns:
        numeric_cols.append("Final_Score")

    for col in numeric_cols:
        if col in df.columns:
            # Strip extra spaces and symbols like '%', '$'
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip().str.replace(r"[%\$,]", "", regex=True)
                df[col] = df[col].replace(["", "nan", "None", "NULL", "null", "NA"], np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Handle out-of-range / invalid values
    invalid_mask = pd.Series(False, index=df.index)
    for col, (min_val, max_val) in VALID_RANGES.items():
        if col in df.columns:
            out_of_bounds = (df[col] < min_val) | (df[col] > max_val)
            cnt = int(out_of_bounds.sum())
            report["invalid_values_detected"][col] = cnt
            if cnt > 0:
                if filter_invalid_rows:
                    invalid_mask = invalid_mask | out_of_bounds
                else:
                    # Clip to valid bounds
                    df[col] = df[col].clip(lower=min_val, upper=max_val)

    if filter_invalid_rows and invalid_mask.sum() > 0:
        df = df[~invalid_mask].reset_index(drop=True)

    # 6. Normalize categorical column: Academic_Participation
    if "Academic_Participation" in df.columns:
        casing_fixes = 0
        def normalize_part(val):
            nonlocal casing_fixes
            if pd.isna(val) or val is None or str(val).strip() in ["", "nan", "None"]:
                return "Medium"  # default mode
            s = str(val).strip().title()
            if s != str(val):
                casing_fixes += 1
            if s not in ["Low", "Medium", "High"]:
                casing_fixes += 1
                return "Medium"
            return s

        df["Academic_Participation"] = df["Academic_Participation"].apply(normalize_part)
        report["casing_inconsistencies_fixed"] = casing_fixes

    # 7. Impute missing values
    # Continuous features: Median imputation
    for col in ["Attendance_Percentage", "Study_Hours", "Previous_Marks", "Assignment_Score", "Internal_Assessment_Marks"]:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(round(median_val, 1))

    # Integer count feature: Median rounded to int
    if "Completed_Assignments" in df.columns and df["Completed_Assignments"].isnull().sum() > 0:
        med_assign = int(round(df["Completed_Assignments"].median()))
        df["Completed_Assignments"] = df["Completed_Assignments"].fillna(med_assign).astype(int)

    # 8. Ensure Performance_Category & Final_Score consistency if present
    if "Final_Score" in df.columns and "Performance_Category" in df.columns:
        # If Final_Score is present, impute any missing from composite formula
        if df["Final_Score"].isnull().sum() > 0:
            approx_score = (
                0.25 * df["Previous_Marks"] +
                0.20 * df["Attendance_Percentage"] +
                0.20 * df["Assignment_Score"] +
                0.25 * (df["Internal_Assessment_Marks"] * 2.0) +
                0.10 * (df["Study_Hours"] * 3.0)
            ).clip(30.0, 99.0).round(1)
            df["Final_Score"] = df["Final_Score"].fillna(approx_score)

        # Derive or re-verify Performance_Category
        def assign_category(score):
            if score < 50.0:
                return "Low"
            elif score < 75.0:
                return "Medium"
            return "High"

        df["Performance_Category"] = df["Final_Score"].apply(assign_category)

    report["cleaned_shape"] = df.shape
    return df, report


def run_cleaning_pipeline(raw_csv_path: Optional[str] = None, output_csv_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs data cleaning from file and persists to processed location."""
    raw_path = raw_csv_path or DEFAULT_RAW_PATH
    out_path = output_csv_path or DEFAULT_CLEAN_PATH

    print(f"Reading raw dataset from: {raw_path}")
    raw_df = pd.read_csv(raw_path)
    clean_df, report = clean_raw_data(raw_df)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    clean_df.to_csv(out_path, index=False)
    print(f"Cleaned dataset saved to: {out_path}")
    print(f"Audit summary: Raw={report['raw_shape']}, Cleaned={report['cleaned_shape']}, Duplicates Removed={report['duplicates_removed']}")

    return clean_df, report


if __name__ == "__main__":
    run_cleaning_pipeline()
