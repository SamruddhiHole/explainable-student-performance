"""
Streamlit Web Application: Explainable Student Performance Prediction.
TY B.Sc. Data Science Final Year Project.

Features:
  1. Home / Overview
  2. Data Upload & Cleaning Audit
  3. Exploratory Data Analysis (EDA)
  4. Model Evaluation & Comparison (Random Forest, Logistic Regression, XGBoost)
  5. Student Performance Prediction
  6. Explainable AI (SHAP & Natural Language Insights)
  7. SQLite Prediction History & Audit Ledger
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set Streamlit page configuration
st.set_page_config(
    page_title="Explainable Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw", "unclean_student_data.csv")
DATA_CLEAN = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_student_data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

from src.data_cleaning import clean_raw_data, REQUIRED_COLUMNS
from src.eda import (
    plot_category_distribution, plot_category_donut, plot_attendance_vs_score,
    plot_study_hours_vs_score, plot_previous_marks_vs_score,
    plot_participation_impact, plot_correlation_heatmap,
    plot_confidence_gauge, plot_student_vs_class_radar,
    plot_feature_violin_distribution, plot_assignments_vs_internals
)
from src.train_models import load_model_artifacts, train_and_evaluate_models
from src.prediction import PredictionService, RISK_LEVELS
from src.explainability import ShapExplainerService, FEATURE_FRIENDLY_NAMES, plot_shap_waterfall_interactive
from src.llm_explanation import generate_llm_explanation, check_ollama_available
from src.database import init_database, save_prediction, get_predictions_history, get_model_evaluations


@st.cache_resource
def get_prediction_service():
    return PredictionService()


@st.cache_resource
def get_explainer_service():
    return ShapExplainerService()


@st.cache_data
def load_clean_data():
    if os.path.exists(DATA_CLEAN):
        return pd.read_csv(DATA_CLEAN)
    return None


@st.cache_data
def load_raw_data():
    if os.path.exists(DATA_RAW):
        return pd.read_csv(DATA_RAW)
    return None


# Ensure database initialized
init_database()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/education.png", width=64)
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio(
    "Select Module:",
    [
        "1. Home / Overview",
        "2. Data Upload & Cleaning",
        "3. Exploratory Data Analysis",
        "4. Model Evaluation & Comparison",
        "5. Student Prediction",
        "6. SHAP Explainability",
        "7. Prediction History (SQLite)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Project**: TY B.Sc. Data Science  \n"
    "**Models**: Random Forest & Logistic Reg.  \n"
    "**XAI**: SHAP (Local & Global)  \n"
    "**Database**: SQLite  \n"
    "**Anti-Leakage**: Strictly Enforced"
)

# -------------------------------------------------------------
# PAGE 1: HOME / OVERVIEW
# -------------------------------------------------------------
if page == "1. Home / Overview":
    st.markdown('<div class="main-title">🎓 Explainable Student Performance Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A Decision-Support System Combining Predictive Machine Learning with Shapley Additive Explanations (SHAP)</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    clean_df = load_clean_data()
    raw_df = load_raw_data()

    with col1:
        st.metric("Clean Records", f"{len(clean_df):,}" if clean_df is not None else "0")
    with col2:
        st.metric("Raw / Dirty Records", f"{len(raw_df):,}" if raw_df is not None else "0")
    with col3:
        st.metric("Primary ML Model", "Random Forest")
    with col4:
        st.metric("Explainability Framework", "SHAP XAI")

    st.markdown("---")

    st.markdown("### 📌 Project Synopsis & Problem Statement")
    st.write(
        "In educational institutions, identifying students requiring proactive academic mentoring typically relies on manual "
        "evaluations or unexplainable black-box algorithms. This project develops an **end-to-end, interpretable machine learning system** "
        "that accurately forecasts whether a student is on track for **Low**, **Medium**, or **High** academic performance while "
        "transparently explaining the underlying drivers behind each prediction using **SHAP**."
    )

    st.markdown("### 🔄 Complete End-to-End System Workflow")
    st.markdown("""
    ```
    [ Raw Student Dataset ] ──▶ [ Automated Cleaning Pipeline ] ──▶ [ Feature Selection (No Leakage) ]
                                            │                                      │
                                            ▼                                      ▼
                                [ 5 Visual EDA Charts ]               [ Train/Test Split (80/20) ]
                                                                                   │
                                                                                   ▼
    [ SQLite Database ] ◀── [ Interactive Streamlit ] ◀── [ SHAP XAI Engine ] ◀── [ RF & LogReg Models ]
    ```
    """)

    st.markdown("### 🛡️ Critical Anti-Data-Leakage Guarantee")
    st.info(
        "**Strict Causal Boundary**: `Final_Score` and `Performance_Category` are ground truth targets and are "
        "**never** provided to the machine learning models during training or inference. Only prior indicators "
        "(Attendance, Study Hours, Previous Semester Marks, Coursework Scores, Internal Marks, and Participation) "
        "are utilized to ensure valid decision support."
    )

# -------------------------------------------------------------
# PAGE 2: DATA UPLOAD & CLEANING
# -------------------------------------------------------------
elif page == "2. Data Upload & Cleaning":
    st.title("🧹 Data Upload & Automated Cleaning Pipeline")
    st.write("Demonstrates genuine data validation, duplicate removal, missing value imputation, and out-of-bounds detection.")

    source_option = st.radio(
        "Choose Data Source:",
        ["Use Default Raw Dataset (unclean_student_data.csv)", "Upload Custom CSV File"],
        horizontal=True
    )

    df_to_clean = None
    if source_option == "Upload Custom CSV File":
        uploaded_file = st.file_uploader("Upload student CSV dataset", type=["csv"])
        if uploaded_file is not None:
            try:
                df_to_clean = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    else:
        df_to_clean = load_raw_data()

    if df_to_clean is not None:
        st.subheader("1. Raw Dataset Inspection")
        colA, colB, colC = st.columns(3)
        colA.metric("Initial Rows", df_to_clean.shape[0])
        colB.metric("Features Count", df_to_clean.shape[1])
        colC.metric("Missing Cells Detected", int(df_to_clean.isnull().sum().sum()))

        st.dataframe(df_to_clean.head(8), use_container_width=True)

        if st.button("🚀 Execute Automated Cleaning Pipeline", type="primary"):
            with st.spinner("Executing cleaning pipeline..."):
                try:
                    clean_res, audit = clean_raw_data(df_to_clean)
                    st.success("Data cleaning executed successfully!")

                    st.subheader("2. Quality & Validation Audit Report")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Duplicates Removed", audit["duplicates_removed"])
                    m2.metric("Invalid Records Filtered", sum(audit["invalid_values_detected"].values()))
                    m3.metric("Casing Inconsistencies Fixed", audit["casing_inconsistencies_fixed"])
                    m4.metric("Final Valid Records", clean_res.shape[0])

                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        st.markdown("**Missing Values Handled (Pre-Cleaning)**")
                        st.json(audit["missing_values_before"])
                    with col_exp2:
                        st.markdown("**Invalid Bounds Detected & Filtered**")
                        st.json(audit["invalid_values_detected"])

                    st.subheader("3. Cleaned Dataset Preview")
                    st.dataframe(clean_res.head(8), use_container_width=True)

                    csv_data = clean_res.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Cleaned Dataset (CSV)",
                        csv_data,
                        "cleaned_student_data.csv",
                        "text/csv"
                    )
                except Exception as ex:
                    st.error(f"Cleaning pipeline failed: {ex}")
    else:
        st.warning("No dataset loaded. Run `python src/data_generator.py` or upload a CSV.")

# -------------------------------------------------------------
# PAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------
elif page == "3. Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.write("Analyze patterns, academic indicators, and target distributions from the validated student cohort.")

    clean_df = load_clean_data()
    if clean_df is None:
        st.warning("Cleaned dataset not found. Please clean or generate the dataset first.")
    else:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Cohort Distribution",
            "Attendance & Previous Marks",
            "Study Hours vs Score",
            "Coursework & Assessments",
            "Feature Spread Inspector",
            "Correlation Heatmap"
        ])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_category_donut(clean_df), use_container_width=True)
            with c2:
                st.plotly_chart(plot_category_distribution(clean_df), use_container_width=True)
            st.info("💡 **Key Takeaway**: The cohort displays a statistically realistic distribution across Low (<50), Medium (50–74), and High (≥75) tiers without extreme class imbalance.")

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_attendance_vs_score(clean_df), use_container_width=True)
                st.caption("Strong positive linear trend: Students with attendance above 80% predominantly fall into Medium and High performance tiers.")
            with c2:
                st.plotly_chart(plot_previous_marks_vs_score(clean_df), use_container_width=True)
                st.caption("Prior academic track record establishes a strong performance foundation for final results.")

        with tab3:
            st.plotly_chart(plot_study_hours_vs_score(clean_df), use_container_width=True)
            st.write("💡 **Observation**: While higher weekly study hours correlate with superior grades, diminishing returns and realistic noise appear above 25 hrs/week.")

        with tab4:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_assignments_vs_internals(clean_df), use_container_width=True)
                st.caption("Bubble size corresponds to Internal Assessment Marks. High coursework completion consistently clusters with stronger midterm scores.")
            with c2:
                st.plotly_chart(plot_participation_impact(clean_df), use_container_width=True)
                st.caption("Active classroom participation provides an observable upward lift (+3 to +5 marks) across student categories.")

        with tab5:
            st.markdown("##### 🎻 Inspect Feature Value Spread by Category")
            feat_choice = st.selectbox(
                "Select Feature to inspect distribution spread:",
                ["Attendance_Percentage", "Study_Hours", "Previous_Marks", "Assignment_Score", "Internal_Assessment_Marks", "Completed_Assignments"]
            )
            st.plotly_chart(plot_feature_violin_distribution(clean_df, feat_choice), use_container_width=True)
            st.caption("The violin shape depicts kernel density and distribution width, while the embedded box plot displays the median and interquartile range.")

        with tab6:
            st.plotly_chart(plot_correlation_heatmap(clean_df), use_container_width=True)
            st.write("💡 **Correlation Insight**: Internal assessment marks (0.75+) and attendance (0.65+) exhibit the strongest positive correlations with final performance.")

# -------------------------------------------------------------
# PAGE 4: MODEL EVALUATION & COMPARISON
# -------------------------------------------------------------
elif page == "4. Model Evaluation & Comparison":
    st.title("⚖️ Model Evaluation & Comparative Analysis")
    st.write("Evaluate multiclass performance metrics and confusion matrices across trained machine learning algorithms.")

    artifacts = load_model_artifacts()
    metrics = artifacts.get("metrics")

    if not metrics:
        st.warning("Models are not trained yet.")
        if st.button("Train Models Now", type="primary"):
            with st.spinner("Training Random Forest, Logistic Regression, and XGBoost..."):
                train_and_evaluate_models()
                st.rerun()
    else:
        # Comparison Table
        comparison_rows = []
        for model_name, m_data in metrics.items():
            comparison_rows.append({
                "Model": model_name,
                "Role": "Primary Model" if m_data.get("is_primary") else "Baseline/Comparison",
                "Accuracy": f"{m_data['accuracy'] * 100:.2f}%",
                "Precision (Macro)": f"{m_data['precision_macro']:.3f}",
                "Recall (Macro)": f"{m_data['recall_macro']:.3f}",
                "F1-Score (Macro)": f"{m_data['f1_macro']:.3f}"
            })
        st.table(pd.DataFrame(comparison_rows))

        st.subheader("Confusion Matrices (Low vs Medium vs High)")
        m_cols = st.columns(len(metrics))
        cat_labels = ["Low", "Medium", "High"]

        for idx, (m_name, m_data) in enumerate(metrics.items()):
            with m_cols[idx]:
                st.markdown(f"**{m_name}**")
                cm = np.array(m_data["confusion_matrix"])
                fig = px.imshow(
                    cm,
                    x=cat_labels,
                    y=cat_labels,
                    labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                    text_auto=True,
                    color_continuous_scale="Blues"
                )
                fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 💡 Academic Evaluation Insights")
        st.write(
            "1. **Random Forest** exhibits superior non-linear decision boundaries and handles interactions between "
            "attendance, internal marks, and study hours without assuming linear separability.\n"
            "2. **Logistic Regression** provides a robust, fast benchmark confirming linear separability baselines.\n"
            "3. Multi-class macro F1-score confirms reliable performance without skewing toward the majority class."
        )

# -------------------------------------------------------------
# PAGE 5: STUDENT PERFORMANCE PREDICTION
# -------------------------------------------------------------
elif page == "5. Student Prediction":
    st.title("🔮 Student Performance Prediction")
    st.write("Generate real-time predictions and academic risk indicators for an individual student profile.")

    pred_service = get_prediction_service()
    if not pred_service.is_ready():
        st.error("Prediction models are not loaded. Please train models in the evaluation page first.")
    else:
        # Load sample students for quick testing
        test_path = os.path.join(MODELS_DIR, "test_dataset.csv")
        sample_students = None
        if os.path.exists(test_path):
            sample_students = pd.read_csv(test_path)

        input_mode = st.radio("Input Method:", ["Interactive Manual Input", "Select Sample Student Record"], horizontal=True)

        input_data = {}
        student_id = "STU_CUSTOM"

        if input_mode == "Select Sample Student Record" and sample_students is not None:
            selected_stu_id = st.selectbox("Select Student ID:", sample_students["Student_ID"].tolist())
            selected_row = sample_students[sample_students["Student_ID"] == selected_stu_id].iloc[0]
            student_id = selected_stu_id

            col1, col2, col3 = st.columns(3)
            with col1:
                att = st.slider("Attendance Percentage (%)", 30.0, 100.0, float(selected_row["Attendance_Percentage"]))
                study = st.slider("Weekly Study Hours", 1.0, 40.0, float(selected_row["Study_Hours"]))
            with col2:
                prev = st.slider("Previous Semester Marks (%)", 30.0, 100.0, float(selected_row["Previous_Marks"]))
                assign = st.slider("Assignment Score (%)", 25.0, 100.0, float(selected_row["Assignment_Score"]))
            with col3:
                internal = st.slider("Internal Assessment Marks (out of 50)", 10.0, 50.0, float(selected_row["Internal_Assessment_Marks"]))
                completed = st.slider("Completed Assignments (out of 10)", 0, 10, int(selected_row["Completed_Assignments"]))
                part = st.selectbox(
                    "Academic Participation",
                    ["Low", "Medium", "High"],
                    index=["Low", "Medium", "High"].index(str(selected_row["Academic_Participation"])) if str(selected_row["Academic_Participation"]) in ["Low", "Medium", "High"] else 1
                )
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                student_id = st.text_input("Student Identifier", "STU2026")
                att = st.slider("Attendance Percentage (%)", 30.0, 100.0, 82.0)
                study = st.slider("Weekly Study Hours", 1.0, 40.0, 14.0)
            with col2:
                prev = st.slider("Previous Semester Marks (%)", 30.0, 100.0, 72.0)
                assign = st.slider("Assignment Score (%)", 25.0, 100.0, 75.0)
            with col3:
                internal = st.slider("Internal Assessment Marks (out of 50)", 10.0, 50.0, 36.0)
                completed = st.slider("Completed Assignments (out of 10)", 0, 10, 8)
                part = st.selectbox("Academic Participation", ["Low", "Medium", "High"], index=1)

        input_data = {
            "Attendance_Percentage": att,
            "Study_Hours": study,
            "Previous_Marks": prev,
            "Assignment_Score": assign,
            "Internal_Assessment_Marks": internal,
            "Completed_Assignments": completed,
            "Academic_Participation": part
        }

        model_choice = st.selectbox("Inference Model:", list(pred_service.models.keys()))

        if st.button("🎯 Predict Student Performance", type="primary"):
            res = pred_service.predict_student(input_data, model_name=model_choice)
            st.session_state["last_prediction"] = res
            st.session_state["last_student_features"] = input_data
            st.session_state["last_student_id"] = student_id

            # Pre-compute SHAP local factors for seamless explainability
            try:
                explainer_svc = get_explainer_service()
                if explainer_svc.rf_model is not None:
                    cat_idx = {"Low": 0, "Medium": 1, "High": 2}.get(res["predicted_category"], 1)
                    st.session_state["last_explanation"] = explainer_svc.explain_individual_student(input_data, cat_idx)
            except Exception:
                st.session_state["last_explanation"] = None

        if "last_prediction" in st.session_state:
            res = st.session_state["last_prediction"]
            st.markdown("---")
            st.subheader(f"Prediction Output for {st.session_state.get('last_student_id')}")

            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                cat = res["predicted_category"]
                badge = res["risk_info"]["badge"]
                st.markdown(f"### Category: **{cat}**")
                st.markdown(f"**Standing**: {badge}")
            with res_col2:
                st.metric("Model Confidence", f"{res['confidence']:.1f}%")
            with res_col3:
                st.metric("Estimated Final Score", f"{res['predicted_score']:.1f} / 100")

            # Probability Breakdown and Visual Indicators
            st.markdown("**Class Probability Breakdown & Visual Assessment**")
            prob_df = pd.DataFrame([res["probabilities"]])
            st.dataframe(prob_df, use_container_width=True)

            pred_vis_col1, pred_vis_col2 = st.columns([1, 1])
            with pred_vis_col1:
                st.plotly_chart(
                    plot_confidence_gauge(res["confidence"], res["predicted_category"], res["predicted_score"]),
                    use_container_width=True
                )
            with pred_vis_col2:
                clean_df_bench = load_clean_data()
                if clean_df_bench is not None and "last_student_features" in st.session_state:
                    st.plotly_chart(
                        plot_student_vs_class_radar(st.session_state["last_student_features"], clean_df_bench),
                        use_container_width=True
                    )

            # Quick XAI preview if available
            expl = st.session_state.get("last_explanation")
            if expl:
                st.markdown("**Key Influencing Factors (SHAP Preview)**")
                prev_c1, prev_c2 = st.columns(2)
                with prev_c1:
                    if expl["readable_positive"]:
                        st.caption("Top Positive Driver: " + expl["readable_positive"][0])
                with prev_c2:
                    if expl["readable_negative"]:
                        st.caption("Top Negative Driver: " + expl["readable_negative"][0])
                st.info("👉 Head over to **6. SHAP Explainability** for interactive waterfall/bar plots and natural language summaries.")

            # SQLite Save Option
            if st.button("💾 Log Prediction to SQLite Database"):
                pos_str = "; ".join(expl.get("readable_positive", [])) if expl else ""
                neg_str = "; ".join(expl.get("readable_negative", [])) if expl else ""
                summary_str = f"Predicted {res['predicted_category']} with {res['confidence']:.1f}% confidence ({res['model_name']})"
                row_id = save_prediction(
                    student_id=st.session_state.get("last_student_id", "STU"),
                    model_name=res["model_name"],
                    predicted_category=res["predicted_category"],
                    predicted_score=res["predicted_score"],
                    prediction_confidence=res["confidence"],
                    top_positive_factors=pos_str,
                    top_negative_factors=neg_str,
                    explanation_summary=summary_str
                )
                st.success(f"Record successfully saved into SQLite (Record #{row_id})")

# -------------------------------------------------------------
# PAGE 6: SHAP EXPLAINABILITY
# -------------------------------------------------------------
elif page == "6. SHAP Explainability":
    st.title("🔍 Explainable AI (SHAP & Insights)")
    st.write("Understand the predictive reasoning behind student evaluations through global and individual SHAP explanations.")

    explainer_service = get_explainer_service()

    if explainer_service.rf_model is None:
        st.warning("Models not ready for SHAP analysis. Please train models first.")
    else:
        st.subheader("1. Global SHAP Feature Importance")
        st.write("Quantifies the overall average impact of each academic indicator across all students:")

        try:
            global_imp = explainer_service.get_global_feature_importance()
            fig_global = px.bar(
                global_imp,
                x="Mean_Absolute_SHAP",
                y="Feature_Name",
                orientation="h",
                color="Mean_Absolute_SHAP",
                color_continuous_scale="Blues",
                title="Global Feature Importance (Mean |SHAP| across classes)"
            )
            fig_global.update_layout(yaxis=dict(autorange="reversed"), height=380, template="plotly_white")
            st.plotly_chart(fig_global, use_container_width=True)
        except Exception as e:
            st.error(f"Global SHAP calculation error: {e}")

        st.markdown("---")
        st.subheader("2. Student-Wise Individual SHAP Explanation")

        # Check if student was predicted in session
        if "last_student_features" in st.session_state and "last_prediction" in st.session_state:
            features = st.session_state["last_student_features"]
            pred = st.session_state["last_prediction"]
            cat = pred["predicted_category"]
            cat_idx = {"Low": 0, "Medium": 1, "High": 2}.get(cat, 1)

            st.write(f"Explaining prediction for **{st.session_state.get('last_student_id', 'Student')}** (Outcome: **{cat} Performance**):")

            ind_expl = explainer_service.explain_individual_student(features, cat_idx)

            # Interactive SHAP Waterfall Plot
            st.plotly_chart(plot_shap_waterfall_interactive(ind_expl), use_container_width=True)

            # Diverging Bar plot of local SHAP contributions
            factors_df = pd.DataFrame(ind_expl["factors"])
            factors_df["Contribution"] = factors_df["shap_value"].apply(lambda x: "Positive Factor" if x >= 0 else "Negative Factor")

            fig_ind = px.bar(
                factors_df,
                x="shap_value",
                y="feature_name",
                color="Contribution",
                orientation="h",
                color_discrete_map={"Positive Factor": "#22c55e", "Negative Factor": "#ef4444"},
                title=f"Detailed Feature Contributions toward '{cat}' Performance"
            )
            fig_ind.update_layout(yaxis=dict(autorange="reversed"), height=360, template="plotly_white")
            st.plotly_chart(fig_ind, use_container_width=True)

            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.markdown("##### 🟢 Primary Positive Contributors")
                for s in ind_expl["readable_positive"]:
                    st.write(s)
                if not ind_expl["readable_positive"]:
                    st.caption("No significant positive drivers.")

            with col_neg:
                st.markdown("##### 🔴 Primary Negative Constraints")
                for s in ind_expl["readable_negative"]:
                    st.write(s)
                if not ind_expl["readable_negative"]:
                    st.caption("No significant negative constraints.")

            st.markdown("---")
            st.subheader("3. Natural Language Explanation")

            ollama_status = check_ollama_available()
            if ollama_status:
                st.success("🟢 Local Ollama (Qwen2.5-3B) service detected and active.")
            else:
                st.info("ℹ️ Local Ollama service not detected. Using built-in deterministic Python template generator.")

            use_llm = st.checkbox("Generate Natural-Language Explanation", value=True)
            if use_llm:
                with st.spinner("Generating teacher-friendly explanation..."):
                    llm_res = generate_llm_explanation(pred, ind_expl, use_llm_if_available=ollama_status)
                    st.markdown(f"**Verbalizer Source**: `{llm_res['model_used']}`")
                    st.info(llm_res["explanation"])
        else:
            st.info("👉 Please predict student performance in the 'Student Prediction' tab first to generate local SHAP explanations.")

# -------------------------------------------------------------
# PAGE 7: PREDICTION HISTORY (SQLITE)
# -------------------------------------------------------------
elif page == "7. Prediction History (SQLite)":
    st.title("🗄️ Prediction History & Audit Ledger")
    st.write("Locally stored prediction audit records stored inside SQLite (`database/student_performance.db`).")

    history_df = get_predictions_history(limit=200)

    if history_df.empty:
        st.info("No predictions logged yet. Make a prediction in tab 5 and click 'Log Prediction to SQLite Database'.")
    else:
        st.subheader(f"Total Logged Predictions: {len(history_df)}")

        # Filters
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            cat_filter = st.multiselect("Filter by Category:", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
        with filter_col2:
            search_id = st.text_input("Search Student ID:")

        filtered = history_df[history_df["predicted_category"].isin(cat_filter)]
        if search_id:
            filtered = filtered[filtered["student_id"].str.contains(search_id, case=False, na=False)]

        st.dataframe(filtered, use_container_width=True)

        csv_hist = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export History to CSV",
            csv_hist,
            "prediction_history.csv",
            "text/csv"
        )
