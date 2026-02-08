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
    auc
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

drop_cols = ["Employee_ID", "Performance_Score", "Resigned"]
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
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=13,
    n_jobs=-1
)

pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", model)
])

# -----------------------------
# Load or Train Model
# -----------------------------
if os.path.exists(MODEL_PATH):
    print("\n✔ Existing model found. Loading model...")
    pipe = joblib.load(MODEL_PATH)
else:
    print("\n⚙ No model found. Training from scratch...")
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODEL_PATH)
    print("✔ Model trained and saved.")

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
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"))
plt.close()

# -----------------------------
# ROC Curve + AUC (Multiclass)
# -----------------------------
y_test_bin = label_binarize(y_test, classes=["Low", "Medium", "High"])

fpr, tpr, roc_auc = {}, {}, {}

plt.figure()

for i, cls in enumerate(["Low", "Medium", "High"]):
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
    for cls, val in zip(["Low", "Medium", "High"], roc_auc.values()):
        f.write(f"{cls} AUC: {val:.4f}\n")

print("\n✔ Reports saved in /reports folder")
