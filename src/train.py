import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data/Extended_Employee_Performance_and_Productivity_Data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "employee_perf_model.pkl")
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

# -----------------------------
# Target Creation (UNCHANGED LOGIC)
# -----------------------------
df["perf_band_next"] = pd.qcut(
    df["Performance_Score"],
    q=3,
    labels=["Low", "Medium", "High"]
)

drop_cols = ["Employee_ID", "Performance_Score", "Resigned", "Hire_Date"]
X = df.drop(columns=drop_cols + ["perf_band_next"], errors="ignore")
y = df["perf_band_next"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=13
)

# -----------------------------
# Preprocessing (UNCHANGED)
# -----------------------------
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=13,
    n_jobs=2
)

pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", model)
])

# -----------------------------
# Load or Train Model
# -----------------------------
if os.path.exists(MODEL_PATH):
    print("\n[OK] Existing model found. Loading model...")
    pipe = joblib.load(MODEL_PATH)
else:
    print("\n[INFO] No model found. Training from scratch...")
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODEL_PATH)
    print("[OK] Model trained and saved.")

# -----------------------------
# Evaluation
# -----------------------------
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)

# Classification Report
report = classification_report(y_test, y_pred, digits=3)
print("\nClassification Report:\n", report)

with open(os.path.join(REPORT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Medium", "High"])

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap="Blues", ax=ax)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"))
plt.close()

# -----------------------------
# Feature Importance
# -----------------------------
print("\n[INFO] Generating Feature Importance...")
clf = pipe.named_steps["clf"]
pre = pipe.named_steps["pre"]

# Handle potential feature name issues with get_feature_names_out
feature_names = pre.get_feature_names_out()
importances = clf.feature_importances_

# Clean names for better visualization
clean_names = [f.replace("num__", "").replace("cat__", "").replace("_", " ").title() for f in feature_names]

imp_df = pd.DataFrame({"Feature": clean_names, "Importance": importances}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(imp_df["Feature"].head(15), imp_df["Importance"].head(15), color="skyblue")
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "feature_importance.png"))
plt.close()

# -----------------------------
# ROC Curve + AUC (Multiclass)
# -----------------------------
classes = pipe.named_steps["clf"].classes_
y_test_bin = label_binarize(y_test, classes=classes)

fpr, tpr, roc_auc = {}, {}, {}

plt.figure(figsize=(10, 8))

for i, cls in enumerate(classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"{cls} (AUC={roc_auc[i]:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(REPORT_DIR, "roc_curve.png"))
plt.close()

with open(os.path.join(REPORT_DIR, "auc_score.txt"), "w") as f:
    for cls, i in zip(classes, range(len(classes))):
        f.write(f"{cls} AUC: {roc_auc[i]:.4f}\n")

print("\n[OK] Reports saved in /reports folder")
