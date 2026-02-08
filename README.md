```markdown
# Employee Performance Predictor

## 📌 Project Overview
The **Employee Performance Predictor** is a machine learning–based application designed to predict employee performance levels using historical employee data.  
The project follows a complete ML pipeline—from data preprocessing and model training to evaluation and deployment using a **Streamlit** web interface.

This system can help HR teams and management make data-driven decisions related to employee productivity and performance assessment.

---

## 🎯 Objectives
- Analyze employee productivity and performance data
- Train a supervised machine learning model for performance prediction
- Evaluate the model using standard classification metrics
- Deploy the trained model as an interactive web application

---

## 🧠 Machine Learning Approach
- **Problem Type:** Classification  
- **Algorithm Used:** Random Forest Classifier  
- **Libraries:** Scikit-learn, Pandas, NumPy  
- **Deployment:** Streamlit  

---

## 📂 Project Structure

```

Employee_Performance_Predictor_Project/
│
├── app/
│   └── app.py (Streamlit web application for prediction)
│
├── data/
│   └── Extended_Employee_Performance_and_Productivity_Data.csv (Main dataset used for training and testing)
│
├── models/
│   └── employee_perf_model.pkl (Trained machine learning model)
│
├── reports/
│   ├── auc_score.txt
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── roc_curve.png
│   └── Model evaluation outputs
│
├── src/
│   └── train.py
│   └── Script for data preprocessing, training, and evaluation
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
└── Test/
└── Sample test resources

````

---

## ⚙️ Project Workflow

### 1️⃣ Data Collection
- Employee performance and productivity data is loaded from CSV files.
- Dataset includes both numerical and categorical features.

### 2️⃣ Data Preprocessing
- Missing values handled using `SimpleImputer`
- Numerical features scaled using `RobustScaler`
- Categorical features encoded using `OneHotEncoder`
- Preprocessing handled via `ColumnTransformer`

### 3️⃣ Model Training
- Data split into training and testing sets
- Random Forest Classifier used for prediction
- Model trained using a Scikit-learn pipeline
- Trained model saved as `.pkl` using `joblib`

### 4️⃣ Model Evaluation
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix
- ROC Curve
- AUC Score
- All evaluation outputs stored in the `reports/` folder

### 5️⃣ Deployment
- Streamlit app loads the trained model
- Users can input employee data via UI
- App predicts employee performance in real time
- Evaluation plots and metrics displayed interactively

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
````

### 2️⃣ Train the Model

```bash
python src/train.py
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app/app.py
```

---

## 📊 Output & Results

* Accurate prediction of employee performance categories
* Visual evaluation metrics (ROC curve, confusion matrix)
* Interactive and user-friendly web interface

---

## 🛠 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* Joblib

---

## 🔮 Future Enhancements

* Add more advanced models (XGBoost, LightGBM)
* Hyperparameter tuning
* Role-based employee prediction
* Integration with live HR databases
* Model explainability using SHAP


## 👤 Author

**Kusan Chakraborty**  
B.Tech – Computer Science & Engineering (Data Science)

---

## 📄 License

This project is licensed under the **MIT License**.

You are free to:
- Use
- Modify
- Distribute

This software, provided proper credit is given to the author.

© 2026 Kusan Chakraborty
