"""
Exploratory Data Analysis (EDA) Module.
Generates interactive Plotly charts examining student academic indicators:
  1. Attendance vs Final Score
  2. Study Hours vs Final Score
  3. Previous Marks vs Final Score
  4. Performance Category Distribution
  5. Academic Participation vs Final Score (Box Plot)
  6. Feature Correlation Heatmap
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

CATEGORY_COLORS = {
    "Low": "#ef4444",     # Red
    "Medium": "#eab308",  # Amber/Yellow
    "High": "#22c55e"     # Green
}


def plot_category_distribution(df: pd.DataFrame) -> go.Figure:
    """Renders distribution of student performance categories."""
    if "Performance_Category" not in df.columns:
        raise ValueError("Missing Performance_Category column.")

    counts = df["Performance_Category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]

    fig = px.bar(
        counts,
        x="Category",
        y="Count",
        color="Category",
        color_discrete_map=CATEGORY_COLORS,
        text="Count",
        title="Student Distribution across Performance Categories"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Performance Category",
        yaxis_title="Number of Students",
        showlegend=False,
        template="plotly_white",
        height=380
    )
    return fig


def plot_attendance_vs_score(df: pd.DataFrame) -> go.Figure:
    """Renders relationship between Class Attendance and Final Score."""
    fig = px.scatter(
        df,
        x="Attendance_Percentage",
        y="Final_Score" if "Final_Score" in df.columns else "Previous_Marks",
        color="Performance_Category",
        color_discrete_map=CATEGORY_COLORS,
        trendline="ols",
        hover_data=["Student_ID", "Study_Hours", "Previous_Marks"] if "Student_ID" in df.columns else None,
        title="Class Attendance (%) vs Final Score"
    )
    fig.update_layout(
        xaxis_title="Attendance Percentage (%)",
        yaxis_title="Final Score (Out of 100)",
        template="plotly_white",
        height=400
    )
    return fig


def plot_study_hours_vs_score(df: pd.DataFrame) -> go.Figure:
    """Renders relationship between Weekly Study Hours and Final Score."""
    fig = px.scatter(
        df,
        x="Study_Hours",
        y="Final_Score" if "Final_Score" in df.columns else "Previous_Marks",
        color="Performance_Category",
        color_discrete_map=CATEGORY_COLORS,
        trendline="ols",
        hover_data=["Student_ID", "Attendance_Percentage"] if "Student_ID" in df.columns else None,
        title="Weekly Study Hours vs Final Score"
    )
    fig.update_layout(
        xaxis_title="Weekly Self-Study Hours",
        yaxis_title="Final Score (Out of 100)",
        template="plotly_white",
        height=400
    )
    return fig


def plot_previous_marks_vs_score(df: pd.DataFrame) -> go.Figure:
    """Renders relationship between Previous Marks and Final Score."""
    fig = px.scatter(
        df,
        x="Previous_Marks",
        y="Final_Score" if "Final_Score" in df.columns else "Assignment_Score",
        color="Performance_Category",
        color_discrete_map=CATEGORY_COLORS,
        trendline="ols",
        hover_data=["Student_ID"] if "Student_ID" in df.columns else None,
        title="Previous Semester Marks (%) vs Final Score"
    )
    fig.update_layout(
        xaxis_title="Previous Semester Marks (%)",
        yaxis_title="Final Score (Out of 100)",
        template="plotly_white",
        height=400
    )
    return fig


def plot_participation_impact(df: pd.DataFrame) -> go.Figure:
    """Box plot of Final Score grouped by Academic Participation."""
    order = ["Low", "Medium", "High"]
    fig = px.box(
        df,
        x="Academic_Participation",
        y="Final_Score" if "Final_Score" in df.columns else "Previous_Marks",
        category_orders={"Academic_Participation": order},
        color="Academic_Participation",
        color_discrete_map={"Low": "#ef4444", "Medium": "#3b82f6", "High": "#22c55e"},
        points="all",
        title="Impact of Academic Participation on Final Score"
    )
    fig.update_layout(
        xaxis_title="Academic Participation Level",
        yaxis_title="Final Score (Out of 100)",
        showlegend=False,
        template="plotly_white",
        height=400
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Renders correlation matrix among numerical features."""
    num_cols = [
        "Attendance_Percentage", "Study_Hours", "Previous_Marks",
        "Assignment_Score", "Internal_Assessment_Marks", "Completed_Assignments"
    ]
    if "Final_Score" in df.columns:
        num_cols.append("Final_Score")

    corr = df[num_cols].corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Feature Correlation Heatmap"
    )
    fig.update_layout(
        template="plotly_white",
        height=450
    )
    return fig
