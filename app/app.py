import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "models/employee_perf_model.pkl"
DATA_PATH = "data/Extended_Employee_Performance_and_Productivity_Data.csv"
REPORT_DIR = "reports"

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Employee Performance Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Corporate Theme CSS
# Ensure text visibility in both Dark and Light mode using rgba and native text colors
# -----------------------------
st.markdown("""
<style>
    /* Corporate Deep Blue styling */
    .stButton>button {
        background-color: #1E3A8A !important;
        color: #ffffff !important;
        border-radius: 5px;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1c3375 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    div[data-testid="metric-container"] {
        background-color: rgba(30, 58, 138, 0.05); /* Very light blue background, works in both dark/light */
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 5% 5% 5% 10%;
        border-radius: 8px;
        border-left: 5px solid #1E3A8A;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

st.title("💼 Employee Performance Predictor")
st.write("Predict **High / Medium / Low** employee performance using ML")

# -----------------------------
# Load Model
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found. Run `python src/train.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# -----------------------------
# Sidebar Navigation
# -----------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "📂 Batch Prediction",
        "👤 Single Employee",
        "📊 Model Insights",
        "📁 Evaluation Reports"
    ]
)

# ======================================================
# PAGE 1 — Batch Prediction
# ======================================================
if page == "📂 Batch Prediction":
    st.subheader("Batch Employee Performance Prediction")

    uploaded = st.file_uploader("Upload employee CSV", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)

        # Columns not used for prediction
        drop_cols = ["Employee_ID", "Performance_Score", "Resigned"]
        X = df.drop(columns=drop_cols, errors="ignore")

        # Predict
        preds = model.predict(X)
        probs = model.predict_proba(X)

        # Save prediction
        df["Predicted_Performance"] = preds

        # 🔑 FIX: correct confidence → class mapping
        class_order = model.classes_   # e.g. ['High', 'Low', 'Medium']

        for i, cls in enumerate(class_order):
            df[f"Confidence_{cls}"] = probs[:, i]

        st.success("✅ Prediction completed")

        # Charts & Dataframe
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("### Prediction Distribution")
            dist = df["Predicted_Performance"].value_counts().reset_index()
            dist.columns = ["Performance Band", "Count"]
            st.bar_chart(dist, x="Performance Band", y="Count", color="#1E3A8A")

        with col2:
            st.write("### Prediction Details")
            # Remove index from UI
            st.dataframe(df.reset_index(drop=True), use_container_width=True, hide_index=True)

        # Download CSV (index already removed)
        st.download_button(
            "⬇ Download Predictions",
            df.to_csv(index=False),
            "employee_predictions.csv",
            "text/csv"
        )

# ======================================================
# PAGE 2 — Single Employee
# ======================================================
elif page == "👤 Single Employee":
    st.subheader("Single Employee Prediction")

    # We need to construct the form directly based on the dataset features
    # Excluding 'Employee_ID', 'Performance_Score', 'Resigned', 'Hire_Date'
    
    with st.form("employee_prediction_form"):
        st.write("### Employee Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Demographics")
            age = st.number_input("Age", min_value=18, max_value=70, value=30)
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
            education = st.selectbox("Education Level", options=["High School", "Bachelor", "Master", "PhD"])
            
        with col2:
            st.markdown("#### Job Details")
            department = st.selectbox("Department", options=["IT", "Sales", "HR", "Finance", "Marketing", "Operations", "Engineering", "Legal", "Customer Support"])
            job_title = st.selectbox("Job Title", options=["Analyst", "Consultant", "Developer", "Engineer", "Manager", "Specialist", "Technician"])
            years_at_company = st.number_input("Years At Company", min_value=0, max_value=50, value=2)
            monthly_salary = st.number_input("Monthly Salary (₹)", min_value=10000, max_value=150000, value=50000, step=5000)
            team_size = st.number_input("Team Size", min_value=1, max_value=100, value=10)

        with col3:
            st.markdown("#### Performance Metrics")
            work_hours = st.number_input("Work Hours / Week", min_value=10, max_value=80, value=40)
            overtime_hours = st.number_input("Overtime Hours", min_value=0, max_value=50, value=5)
            projects_handled = st.number_input("Projects Handled", min_value=0, max_value=100, value=10)
            training_hours = st.number_input("Training Hours", min_value=0, max_value=200, value=20)
            sick_days = st.number_input("Sick Days", min_value=0, max_value=50, value=2)
            remote_freq = st.number_input("Remote Work Freq (0-10)", min_value=0, max_value=10, value=0)
            promotions = st.number_input("Promotions", min_value=0, max_value=10, value=0)
            satisfaction = st.slider("Satisfaction Score", min_value=1.0, max_value=5.0, value=3.5, step=0.1)

        submitted = st.form_submit_button("Predict Performance")

    if submitted:
        input_data = {
            "Department": department,
            "Gender": gender,
            "Age": age,
            "Job_Title": job_title,
            "Years_At_Company": years_at_company,
            "Education_Level": education,
            "Monthly_Salary": monthly_salary,
            "Work_Hours_Per_Week": work_hours,
            "Projects_Handled": projects_handled,
            "Overtime_Hours": overtime_hours,
            "Sick_Days": sick_days,
            "Remote_Work_Frequency": remote_freq,
            "Team_Size": team_size,
            "Training_Hours": training_hours,
            "Promotions": promotions,
            "Employee_Satisfaction_Score": satisfaction
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Predict
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.markdown("---")
        st.subheader(f"🎯 Predicted Performance: **{pred}**")

        # FIX: correct class-probability mapping
        class_order = model.classes_
        prob_map = dict(zip(class_order, proba))

        display_order = ["High", "Medium", "Low"]
        
        st.write("#### Prediction Confidence")
        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        
        for idx, cls in enumerate(display_order):
            if cls in prob_map:
                cols[idx].metric(label=f"{cls} Confidence", value=f"{prob_map[cls] * 100:.1f}%")


# ======================================================
# PAGE 3 — Model Insights
# ======================================================
elif page == "📊 Model Insights":
    st.subheader("Global Feature Importance")

    clf = model.named_steps["clf"]
    pre = model.named_steps["pre"]

    feature_names = pre.get_feature_names_out()
    importances = clf.feature_importances_

    def clean_feature_name(name):
        name = name.replace("num__", "")
        name = name.replace("cat__", "")
        name = name.replace("_", " ")
        name = name.title()
        return name
    clean_names = [clean_feature_name(f) for f in feature_names]

    imp_df = pd.DataFrame({
        "Feature": clean_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(20)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(imp_df, use_container_width=True, hide_index=True)
    with col2:
        fi_path = os.path.join(REPORT_DIR, "feature_importance.png")
        if os.path.exists(fi_path):
            st.image(Image.open(fi_path), caption="Top Feature Importances", use_container_width=True)

    st.info("These features have the strongest influence on employee performance predictions.")

# ======================================================
# PAGE 4 — Evaluation Reports
# ======================================================
else:
    st.subheader("Model Evaluation Reports")

    if not os.path.exists(REPORT_DIR):
        st.warning("No reports found. Run training first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            cm_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(Image.open(cm_path), caption="Confusion Matrix")

        with col2:
            roc_path = os.path.join(REPORT_DIR, "roc_curve.png")
            if os.path.exists(roc_path):
                st.image(Image.open(roc_path), caption="ROC Curve")

        rep_path = os.path.join(REPORT_DIR, "classification_report.txt")
        if os.path.exists(rep_path):
            st.subheader("Classification Report")
            with open(rep_path) as f:
                st.code(f.read(), language="text")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("ML-based Employee Performance Prediction System | Production-ready")
