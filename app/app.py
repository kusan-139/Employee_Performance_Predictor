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
    layout="wide"
)

st.title("Employee Performance Predictor")
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

        # Remove index from UI
        st.dataframe(df.reset_index(drop=True), use_container_width=True,hide_index=True)

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

    template = pd.read_csv(DATA_PATH).drop(
        columns=["Employee_ID", "Performance_Score", "Resigned"],
        errors="ignore"
    )

    # Fix Hire_Date format
    if "Hire_Date" in template.columns:
        template["Hire_Date"] = pd.to_datetime(
            template["Hire_Date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    input_df = st.data_editor(
        template.head(1),
        use_container_width=True,
        num_rows="fixed",
        hide_index=True
    )

    if st.button("Predict"):
        # Predict
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.subheader(f"🎯 Predicted Performance: **{pred}**")

        # 🔑 FIX: correct class-probability mapping
        class_order = model.classes_   # e.g. ['High', 'Low', 'Medium']
        prob_map = dict(zip(class_order, proba))

        # 🔽 Sort display order (High → Medium → Low)
        display_order = ["High", "Medium", "Low"]

        st.write("Prediction Confidence")
        for cls in display_order:
            if cls in prob_map:
                st.metric(
                    label=cls,
                    value=f"{prob_map[cls] * 100:.2f}%"
                )


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
    st.dataframe(imp_df, use_container_width=True,hide_index=True)

    st.info(
        "These features have the strongest influence on employee performance predictions."
    )

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
