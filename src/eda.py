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

try:
    import statsmodels.api
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

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
        trendline="ols" if STATSMODELS_AVAILABLE else None,
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
        trendline="ols" if STATSMODELS_AVAILABLE else None,
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
        trendline="ols" if STATSMODELS_AVAILABLE else None,
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


def plot_category_donut(df: pd.DataFrame) -> go.Figure:
    """Renders a polished donut chart of performance categories with count & percentage."""
    if "Performance_Category" not in df.columns:
        raise ValueError("Missing Performance_Category column.")

    counts = df["Performance_Category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]

    total = counts["Count"].sum()
    fig = px.pie(
        counts,
        values="Count",
        names="Category",
        hole=0.55,
        color="Category",
        color_discrete_map=CATEGORY_COLORS,
        title="Performance Cohort Breakdown (Donut Distribution)"
    )
    fig.update_traces(
        textposition="outside",
        textinfo="percent+label",
        hoverinfo="label+value+percent",
        marker=dict(line=dict(color="#ffffff", width=2))
    )
    fig.update_layout(
        annotations=[dict(text=f"Total<br><b>{total}</b>", x=0.5, y=0.5, font_size=16, showarrow=False)],
        template="plotly_white",
        height=380,
        showlegend=True
    )
    return fig


def plot_confidence_gauge(confidence_pct: float, category: str, predicted_score: float = None) -> go.Figure:
    """Renders an interactive speedometer gauge for model confidence."""
    bar_color = CATEGORY_COLORS.get(category, "#3b82f6")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_pct,
        number={"suffix": "%", "font": {"size": 34, "color": bar_color}},
        title={"text": f"<b>Prediction Confidence</b><br><span style='font-size:14px;color:gray'>Outcome: {category} Standing</span>", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9ca3af"},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "#f3f4f6",
            "borderwidth": 1,
            "bordercolor": "#e5e7eb",
            "steps": [
                {"range": [0, 50], "color": "#fee2e2"},
                {"range": [50, 75], "color": "#fef3c7"},
                {"range": [75, 100], "color": "#dcfce7"}
            ],
            "threshold": {
                "line": {"color": "#1e3a8a", "width": 4},
                "thickness": 0.8,
                "value": confidence_pct
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=55, b=20),
        template="plotly_white"
    )
    return fig


def plot_student_vs_class_radar(student_features: dict, class_df: pd.DataFrame) -> go.Figure:
    """
    Renders a radar (spider) chart comparing a single student's normalized
    academic profile against the cohort class average.
    """
    categories = [
        "Attendance (%)",
        "Study Hours",
        "Previous Marks (%)",
        "Assignment Score (%)",
        "Internal Marks (scaled %)",
        "Assignments Done (scaled %)"
    ]

    # Normalize metrics to 0-100 scale for intuitive visual comparison
    stu_vals = [
        float(student_features.get("Attendance_Percentage", 0)),
        min(100.0, float(student_features.get("Study_Hours", 0)) * 3.33),
        float(student_features.get("Previous_Marks", 0)),
        float(student_features.get("Assignment_Score", 0)),
        min(100.0, float(student_features.get("Internal_Assessment_Marks", 0)) * 2.0),
        min(100.0, float(student_features.get("Completed_Assignments", 0)) * 10.0)
    ]

    # Calculate class averages
    class_vals = [
        float(class_df["Attendance_Percentage"].mean()),
        min(100.0, float(class_df["Study_Hours"].mean()) * 3.33),
        float(class_df["Previous_Marks"].mean()),
        float(class_df["Assignment_Score"].mean()),
        min(100.0, float(class_df["Internal_Assessment_Marks"].mean()) * 2.0),
        min(100.0, float(class_df["Completed_Assignments"].mean()) * 10.0)
    ]

    # Close the radar loop
    stu_vals += [stu_vals[0]]
    class_vals += [class_vals[0]]
    cat_loop = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=class_vals,
        theta=cat_loop,
        fill="toself",
        fillcolor="rgba(156, 163, 175, 0.25)",
        line=dict(color="#6b7280", dash="dash", width=2),
        name="Cohort Average"
    ))
    fig.add_trace(go.Scatterpolar(
        r=stu_vals,
        theta=cat_loop,
        fill="toself",
        fillcolor="rgba(59, 130, 246, 0.4)",
        line=dict(color="#2563eb", width=3),
        name="Selected Student"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10))
        ),
        showlegend=True,
        title="Student Profile vs Cohort Average (360° Radar)",
        height=360,
        margin=dict(l=35, r=35, t=45, b=25),
        template="plotly_white"
    )
    return fig


def plot_feature_violin_distribution(df: pd.DataFrame, feature_col: str) -> go.Figure:
    """Renders a violin plot showing the distribution spread across performance classes."""
    friendly_name = feature_col.replace("_", " ")
    fig = px.violin(
        df,
        x="Performance_Category",
        y=feature_col,
        color="Performance_Category",
        color_discrete_map=CATEGORY_COLORS,
        box=True,
        points="outliers",
        title=f"Distribution Spread of {friendly_name} by Category"
    )
    fig.update_layout(
        xaxis_title="Performance Category",
        yaxis_title=friendly_name,
        showlegend=False,
        template="plotly_white",
        height=390
    )
    return fig


def plot_assignments_vs_internals(df: pd.DataFrame) -> go.Figure:
    """Renders bubble chart of Completed Assignments vs Assignment Score, sized by Internal Marks."""
    fig = px.scatter(
        df,
        x="Completed_Assignments",
        y="Assignment_Score",
        size="Internal_Assessment_Marks",
        color="Performance_Category",
        color_discrete_map=CATEGORY_COLORS,
        hover_data=["Student_ID", "Attendance_Percentage"] if "Student_ID" in df.columns else None,
        title="Assignments Completed vs Score (Bubble Size = Internal Marks)"
    )
    fig.update_layout(
        xaxis_title="Assignments Completed (0 to 10)",
        yaxis_title="Coursework Assignment Score (%)",
        template="plotly_white",
        height=400
    )
    return fig

