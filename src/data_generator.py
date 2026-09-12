"""
Data Generator module for Explainable Student Performance Prediction.
Generates:
1. Realistic clean dataset (~1,000 records) with natural noise and correlations.
2. Raw unclean dataset (~1,200 records) exhibiting realistic data anomalies
   (missing values, duplicates, out-of-range records, casing inconsistencies, string numbers).
"""

import os
import random
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "unclean_student_data.csv")
CLEAN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_student_data.csv")


def generate_clean_data(n_samples: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a statistically plausible, realistic clean dataset.
    Features:
      - Student_ID
      - Attendance_Percentage (35.0 - 100.0)
      - Study_Hours (1.0 - 35.0)
      - Previous_Marks (30.0 - 98.0)
      - Assignment_Score (25.0 - 100.0)
      - Internal_Assessment_Marks (10.0 - 50.0)
      - Completed_Assignments (0 - 10)
      - Academic_Participation ('Low', 'Medium', 'High')
    Targets:
      - Final_Score (derived from predictors with non-linear relationships & realistic noise)
      - Performance_Category ('Low', 'Medium', 'High')
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Base academic capability latent variable (normal distribution)
    latent_ability = np.random.normal(0, 1, n_samples)

    # 1. Previous Marks: heavily influenced by ability + noise
    prev_marks = 65 + 15 * latent_ability + np.random.normal(0, 6, n_samples)
    prev_marks = np.clip(prev_marks, 30.0, 98.0).round(1)

    # 2. Attendance Percentage: moderately influenced by ability + noise
    attendance = 72 + 12 * latent_ability + np.random.normal(0, 8, n_samples)
    attendance = np.clip(attendance, 38.0, 100.0).round(1)

    # 3. Study Hours: weekly self-study hours
    study_hours = 14 + 5 * latent_ability + np.random.normal(0, 4, n_samples)
    study_hours = np.clip(study_hours, 2.0, 32.0).round(1)

    # 4. Completed Assignments (0 - 10)
    assign_latent = 7.0 + 1.8 * latent_ability + 0.03 * (attendance - 70) + np.random.normal(0, 1.2, n_samples)
    completed_assignments = np.clip(np.round(assign_latent), 0, 10).astype(int)

    # 5. Assignment Score: depends on completed assignments, study hours & ability
    assign_score = 45 + 3.5 * completed_assignments + 0.8 * study_hours + 8 * latent_ability + np.random.normal(0, 7, n_samples)
    assign_score = np.clip(assign_score, 30.0, 100.0).round(1)

    # 6. Internal Assessment Marks (out of 50)
    internal_marks = 28 + 7 * latent_ability + 0.12 * prev_marks + np.random.normal(0, 4, n_samples)
    internal_marks = np.clip(internal_marks, 10.0, 50.0).round(1)

    # 7. Academic Participation ('Low', 'Medium', 'High')
    participation_scores = 0.4 * (attendance / 100.0) + 0.3 * (completed_assignments / 10.0) + 0.3 * np.random.uniform(0, 1, n_samples)
    participation = []
    for p in participation_scores:
        if p < 0.45:
            participation.append("Low")
        elif p < 0.75:
            participation.append("Medium")
        else:
            participation.append("High")

    # Map participation to numerical bonus for final score
    part_map = {"Low": -3.5, "Medium": 1.5, "High": 5.0}
    part_effect = np.array([part_map[p] for p in participation])

    # 8. Final Score Calculation: Realistic weighted composite + realistic human variation/noise
    # Weights reflect typical Indian university grading scheme:
    # 25% Previous Marks + 20% Attendance + 20% Assignment Score + 25% Internal (scaled to 100) + 10% Study Hours + Participation
    internal_scaled = internal_marks * 2.0  # scale 50 -> 100
    study_scaled = np.clip(study_hours * 3.0, 0, 100)

    composite_score = (
        0.25 * prev_marks +
        0.20 * attendance +
        0.20 * assign_score +
        0.25 * internal_scaled +
        0.10 * study_scaled +
        part_effect +
        np.random.normal(0, 5.5, n_samples)  # realistic noise to avoid synthetic 100% accuracy
    )
    final_score = np.clip(composite_score, 30.0, 99.0).round(1)

    # 9. Performance Category:
    # Low: < 50
    # Medium: 50 <= score < 75
    # High: >= 75
    categories = []
    for s in final_score:
        if s < 50.0:
            categories.append("Low")
        elif s < 75.0:
            categories.append("Medium")
        else:
            categories.append("High")

    student_ids = [f"STU{1001 + i}" for i in range(n_samples)]

    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Attendance_Percentage": attendance,
        "Study_Hours": study_hours,
        "Previous_Marks": prev_marks,
        "Assignment_Score": assign_score,
        "Internal_Assessment_Marks": internal_marks,
        "Completed_Assignments": completed_assignments,
        "Academic_Participation": participation,
        "Final_Score": final_score,
        "Performance_Category": categories
    })

    return df


def generate_unclean_data(clean_df: pd.DataFrame, target_raw_size: int = 1200, random_seed: int = 101) -> pd.DataFrame:
    """
    Creates an unclean dataset from base records by injecting realistic data-quality flaws:
      - Duplicate records (~40 rows)
      - Missing values (NaN, None, blanks) in various columns
      - Inconsistent categorical representations (e.g. 'HIGH', 'high', 'Low ', ' MEDIUM')
      - Numbers formatted as strings (e.g. '75.5', ' 82.0 ', '$42.0')
      - Out-of-bounds/invalid records (Attendance > 100%, negative study hours, internal marks > 50)
      - Blank student IDs or whitespace padding
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Start with a copy of clean records
    df = clean_df.copy()

    # 1. Add extra rows to reach target_raw_size
    extra_needed = target_raw_size - len(df)
    if extra_needed > 0:
        # Sample random rows to simulate duplicate records
        duplicates = df.sample(n=extra_needed, replace=True, random_state=random_seed).copy()
        df = pd.concat([df, duplicates], ignore_index=True)

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    n_rows = len(df)

    # 2. Inject casing and whitespace inconsistencies into Academic_Participation
    def corrupt_participation(val):
        choice = random.random()
        if choice < 0.20:
            return val.upper()  # 'HIGH', 'LOW', 'MEDIUM'
        elif choice < 0.40:
            return val.lower()  # 'high', 'low', 'medium'
        elif choice < 0.50:
            return f" {val} "  # extra padding
        elif choice < 0.55:
            return f"{val.lower()} "
        return val

    df["Academic_Participation"] = df["Academic_Participation"].apply(corrupt_participation)

    # 3. Inject string numbers with formatting issues
    study_col = df["Study_Hours"].astype(object).copy()
    prev_col = df["Previous_Marks"].astype(object).copy()
    for idx in range(n_rows):
        r = random.random()
        if r < 0.05:
            study_col.iloc[idx] = f" {study_col.iloc[idx]} "
        elif r < 0.08:
            study_col.iloc[idx] = f"{study_col.iloc[idx]}"

        r2 = random.random()
        if r2 < 0.04:
            prev_col.iloc[idx] = f"{prev_col.iloc[idx]}%"
        elif r2 < 0.08:
            prev_col.iloc[idx] = f" {prev_col.iloc[idx]} "

    df["Study_Hours"] = study_col
    df["Previous_Marks"] = prev_col

    # 4. Inject out-of-bounds/invalid values in ~3-4% of records
    invalid_indices = random.sample(range(n_rows), int(n_rows * 0.04))
    for idx in invalid_indices:
        flaw_type = random.choice(["att_high", "att_neg", "study_neg", "internal_high", "assign_high"])
        if flaw_type == "att_high":
            df.at[idx, "Attendance_Percentage"] = random.choice([105.0, 118.5, 125.0])
        elif flaw_type == "att_neg":
            df.at[idx, "Attendance_Percentage"] = -8.0
        elif flaw_type == "study_neg":
            df.at[idx, "Study_Hours"] = -3.5
        elif flaw_type == "internal_high":
            df.at[idx, "Internal_Assessment_Marks"] = random.choice([65.0, 85.0, 99.0])  # max is 50
        elif flaw_type == "assign_high":
            df.at[idx, "Completed_Assignments"] = random.choice([15, 20, -2])

    # 5. Inject Missing / NaN values (~5% across features)
    feature_cols = [
        "Attendance_Percentage", "Study_Hours", "Previous_Marks",
        "Assignment_Score", "Internal_Assessment_Marks",
        "Completed_Assignments", "Academic_Participation"
    ]
    # Ensure columns can accept mixed string/nan values as in raw CSVs
    for col in feature_cols:
        df[col] = df[col].astype(object)

    for col in feature_cols:
        missing_idx = random.sample(range(n_rows), int(n_rows * 0.035))
        for idx in missing_idx:
            if random.random() < 0.7:
                df.at[idx, col] = np.nan
            else:
                df.at[idx, col] = ""

    return df


def generate_and_save_datasets():
    """Generates both clean and raw datasets and writes them to disk."""
    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)

    print("Generating clean student dataset (1,000 records)...")
    clean_df = generate_clean_data(n_samples=1000, random_seed=42)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Clean dataset saved: {CLEAN_DATA_PATH} (Shape: {clean_df.shape})")

    print("Generating unclean student dataset (~1,200 records)...")
    raw_df = generate_unclean_data(clean_df, target_raw_size=1200, random_seed=101)
    raw_df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Unclean raw dataset saved: {RAW_DATA_PATH} (Shape: {raw_df.shape})")

    return clean_df, raw_df


if __name__ == "__main__":
    generate_and_save_datasets()
