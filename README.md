# Employee Performance Predictor

## 📌 Project Overview
The **Employee Performance Predictor** is an end-to-end machine learning application that predicts employee performance levels based on historical productivity and performance data.  
The project covers the complete ML lifecycle—from data preprocessing and model training to evaluation and deployment using a **Streamlit** web application.

This system helps HR teams and management make **data-driven decisions** related to employee performance analysis.

---

## 🎯 Objectives
- Analyze employee productivity and performance data
- Build a supervised machine learning model for performance prediction
- Evaluate the model using standard classification metrics
- Deploy the trained model as an interactive web application

---

## 🧠 Machine Learning Approach
- **Problem Type:** Classification  
- **Algorithm Used:** Random Forest Classifier  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Model Persistence:** Joblib  
- **Deployment:** Streamlit  

---

## 📂 Project Structure

```text
Employee_Performance_Predictor_Project/
│
├── app/
│   └── app.py
│       └── Streamlit web application for prediction
│
├── data/
│   └── Extended_Employee_Performance_and_Productivity_Data.csv
│       └── Main dataset used for training and testing
│
├── models/
│   └── employee_perf_model.pkl
│       └── Trained machine learning model
│
├── reports/
│   ├── auc_score.txt
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── roc_curve.png
│       └── Model evaluation outputs
│
├── src/
│   └── train.py
│       └── Data preprocessing, training, and evaluation script
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
└── Test/
    └── Sample test resources
```
---

## ⚙️ Project Workflow

### 1️⃣ Data Collection

* Employee performance and productivity data is loaded from CSV files.
* Dataset contains both numerical and categorical features.

### 2️⃣ Data Preprocessing

* Handling missing values using `SimpleImputer`
* Scaling numerical features using `RobustScaler`
* Encoding categorical features using `OneHotEncoder`
* Preprocessing implemented using `ColumnTransformer`

### 3️⃣ Model Training

* Dataset split into training and testing sets
* Random Forest Classifier trained using a Scikit-learn pipeline
* Trained model saved using `joblib` as a `.pkl` file

### 4️⃣ Model Evaluation

* Classification Report (Precision, Recall, F1-score)
* Confusion Matrix
* ROC Curve
* AUC Score
* Evaluation results stored in the `reports/` directory

### 5️⃣ Deployment

* Streamlit application loads the trained model
* User inputs employee data via the web interface
* Application predicts employee performance in real time

---

## 🏋️ How to Train the Model

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Dataset Availability**

   * Place the dataset in the `data/` directory:

     ```
     Extended_Employee_Performance_and_Productivity_Data.csv
     ```

3. **Run the Training Script**

   ```bash
   python src/train.py
   ```

4. **Training Output**

   * Trained model saved in `models/`
   * Evaluation reports generated in `reports/`
   * Model ready for deployment in the Streamlit app

### 🔄 Retraining the Model

To retrain the model with updated data:

* Replace the dataset in the `data/` directory
* Re-run:

  ```bash
  python src/train.py
  ```

---

## 🚀 How to Run the Application

```bash
python -m streamlit run app/app.py
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

* Use advanced models (XGBoost, LightGBM)
* Hyperparameter tuning
* Role-specific performance prediction
* Integration with real-time HR databases
* Model explainability using SHAP

---

## 👤 Author

**Kusan Chakraborty**
B.Tech – Computer Science & Engineering (Data Science)

---

## 📄 License

This project is licensed under the **MIT License**.

You are free to:

* Use
* Modify
* Distribute

This software, provided proper credit is given to the author.

© 2026 Kusan Chakraborty


