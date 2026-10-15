"""
Exploratory Data Analysis (EDA) Module.
Provides simple, intuitive, and visually attractive Plotly charts for student performance:
  1. Executive Cohort Breakdown (Donut Chart)
  2. Attendance vs Final Score Brackets (Bar Chart)
  3. Study Hours Impact on Marks (Bar Chart)
  4. Top Success Factors Ranking (Horizontal Bar Chart)
  + Gauge and Radar charts for individual student prediction.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

CATEGORY_COLORS = {
    "Low": "#ef4444",     # Red - At Risk
    "Medium": "#f59e0b",  # Amber/Yellow - Average
    "High": "#10b981"     # Emerald Green - Distinction
}


# ============================================================================
# 1. SIMPLE EXECUTIVE COHORT BREAKDOWN (DONUT CHART)
# ============================================================================
def plot_simple_cohort_donut(df: pd.DataFrame) -> go.Figure:
    """
    Renders an elegant, simple Donut chart showing the distribution of
    High, Medium, and Low performing students with clear percentages and counts.
    """
    if "Performance_Category" not in df.columns:
        raise ValueError("Missing Performance_Category column.")

    counts = df["Performance_Category"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
    counts.columns = ["Category", "Count"]
    total = int(counts["Count"].sum())

    fig = px.pie(
        counts,
        values="Count",
        names="Category",
        hole=0.55,
        color="Category",
        color_discrete_map=CATEGORY_COLORS,
        title="Student Performance Breakdown (Overall Cohort)"
    )
    fig.update_traces(
        textposition="outside",
        textinfo="label+percent+value",
        hoverinfo="label+value+percent",
        marker=dict(line=dict(color="#ffffff", width=2.5))
    )
    fig.update_layout(
        annotations=[dict(
            text=f"Total<br><b>{total:,}</b><br><span style='font-size:12px;color:gray'>Students</span>",
            x=0.5, y=0.5,
            font_size=15,
            showarrow=False
        )],
        template="plotly_white",
        height=380,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=30)
    )
    return fig


# ============================================================================
# 2. ATTENDANCE BRACKETS VS AVERAGE MARKS (BAR CHART)
# ============================================================================
def plot_attendance_pass_rate_bars(df: pd.DataFrame) -> go.Figure:
    """
    Renders an easy-to-understand bar chart showing how attendance tiers
    directly impact average final marks (Out of 100).
    """
    df_temp = df.copy()

    def get_bracket(att):
        if att < 60:
            return "Below 60% (Critical Risk ❌)"
        elif att < 75:
            return "60% to 74% (Needs Attention ⚠️)"
        elif att < 85:
            return "75% to 84% (Safe Zone ✅)"
        else:
            return "85% & Above (Distinction 🌟)"

    df_temp["Bracket"] = df_temp["Attendance_Percentage"].apply(get_bracket)

    order = [
        "Below 60% (Critical Risk ❌)",
        "60% to 74% (Needs Attention ⚠️)",
        "75% to 84% (Safe Zone ✅)",
        "85% & Above (Distinction 🌟)"
    ]

    grouped = df_temp.groupby("Bracket", observed=False)["Final_Score"].agg(["mean", "count"]).reindex(order).reset_index()
    grouped["mean"] = grouped["mean"].round(1)

    colors = ["#ef4444", "#f59e0b", "#3b82f6", "#10b981"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grouped["Bracket"],
        y=grouped["mean"],
        text=[f"<b>{val:.1f} Marks</b><br>({cnt} students)" for val, cnt in zip(grouped["mean"], grouped["count"])],
        textposition="outside",
        marker=dict(color=colors, line=dict(color="#ffffff", width=1.5)),
        hoverinfo="x+y"
    ))

    # Add passing benchmark line at 50 marks
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text="Passing Threshold (50 Marks)",
        annotation_position="bottom right"
    )

    fig.update_layout(
        title="Class Attendance vs Average Final Marks",
        xaxis_title="Attendance Bracket",
        yaxis_title="Average Final Marks (Out of 100)",
        yaxis_range=[0, 105],
        template="plotly_white",
        height=380,
        margin=dict(l=30, r=30, t=50, b=40)
    )
    return fig


# ============================================================================
# 3. WEEKLY STUDY HOURS IMPACT (BAR CHART)
# ============================================================================
def plot_study_hours_impact_bars(df: pd.DataFrame) -> go.Figure:
    """
    Renders an intuitive bar chart demonstrating how weekly self-study hours
    translate into higher academic marks.
    """
    df_temp = df.copy()

    def get_study_bracket(hrs):
        if hrs < 10:
            return "< 10 Hours / Week<br>(Low Effort)"
        elif hrs <= 20:
            return "10 to 20 Hours / Week<br>(Regular Habit)"
        else:
            return "> 20 Hours / Week<br>(Dedicated Study)"

    df_temp["Study_Bracket"] = df_temp["Study_Hours"].apply(get_study_bracket)

    order = [
        "< 10 Hours / Week<br>(Low Effort)",
        "10 to 20 Hours / Week<br>(Regular Habit)",
        "> 20 Hours / Week<br>(Dedicated Study)"
    ]

    grouped = df_temp.groupby("Study_Bracket", observed=False)["Final_Score"].agg(["mean", "count"]).reindex(order).reset_index()
    grouped["mean"] = grouped["mean"].round(1)

    colors = ["#f87171", "#60a5fa", "#34d399"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grouped["Study_Bracket"],
        y=grouped["mean"],
        text=[f"<b>{val:.1f} Marks</b><br>({cnt} students)" for val, cnt in zip(grouped["mean"], grouped["count"])],
        textposition="outside",
        marker=dict(color=colors, line=dict(color="#ffffff", width=1.5)),
        hoverinfo="x+y"
    ))

    fig.update_layout(
        title="Weekly Self-Study Hours vs Average Final Marks",
        xaxis_title="Study Hours Bracket",
        yaxis_title="Average Final Marks (Out of 100)",
        yaxis_range=[0, 105],
        template="plotly_white",
        height=380,
        margin=dict(l=30, r=30, t=50, b=40)
    )
    return fig


# ============================================================================
# 4. TOP SUCCESS FACTORS (RANKED HORIZONTAL BAR CHART)
# ============================================================================
def plot_top_success_factors(df: pd.DataFrame) -> go.Figure:
    """
    Renders a clear, ranked horizontal bar chart showing the top academic factors
    that determine a student's final score (based on correlation analysis).
    """
    factors = [
        ("Internal Assessment Marks", "Internal_Assessment_Marks", "Continuous exam tests"),
        ("Class Attendance", "Attendance_Percentage", "Regular lecture presence"),
        ("Coursework Assignments", "Assignment_Score", "Practical homework score"),
        ("Previous Semester Marks", "Previous_Marks", "Past academic foundation"),
        ("Weekly Self-Study", "Study_Hours", "Personal revision hours")
    ]

    records = []
    target_col = "Final_Score" if "Final_Score" in df.columns else "Previous_Marks"

    for label, col, desc in factors:
        if col in df.columns and target_col in df.columns:
            corr_val = float(df[col].corr(df[target_col]))
            records.append({
                "Factor": label,
                "Correlation": round(corr_val * 100, 1),
                "Description": desc
            })

    summary_df = pd.DataFrame(records).sort_values("Correlation", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=summary_df["Factor"],
        x=summary_df["Correlation"],
        orientation="h",
        text=[f"<b>{val:.0f}% Impact</b>" for val in summary_df["Correlation"]],
        textposition="inside",
        marker=dict(
            color=summary_df["Correlation"],
            colorscale="Teal",
            line=dict(color="#ffffff", width=1.5)
        ),
        hovertext=[f"{f}: {c}% correlation with final score" for f, c in zip(summary_df["Factor"], summary_df["Correlation"])],
        hoverinfo="text"
    ))

    fig.update_layout(
        title="Top 5 Factors Leading to High Student Grades",
        xaxis_title="Relative Impact on Final Marks (%)",
        yaxis_title="Academic Indicator",
        xaxis_range=[0, 100],
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=30, t=50, b=40)
    )
    return fig


# ============================================================================
# 5. INDIVIDUAL PREDICTION VISUALIZATIONS (RADAR & GAUGE)
# ============================================================================
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

    class_vals = [
        float(class_df["Attendance_Percentage"].mean()),
        min(100.0, float(class_df["Study_Hours"].mean()) * 3.33),
        float(class_df["Previous_Marks"].mean()),
        float(class_df["Assignment_Score"].mean()),
        min(100.0, float(class_df["Internal_Assessment_Marks"].mean()) * 2.0),
        min(100.0, float(class_df["Completed_Assignments"].mean()) * 10.0)
    ]

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


# ============================================================================
# BACKWARD-COMPATIBILITY ALIASES (Ensures no imports break)
# ============================================================================
plot_category_donut = plot_simple_cohort_donut
plot_category_distribution = plot_simple_cohort_donut
plot_attendance_tiers_success_rate = plot_attendance_pass_rate_bars
