# auto-loan-default-prediction-system-classification


## 📌 Project Overview
This project aims to predict the likelihood of a borrower defaulting on an auto loan using historical loan application and repayment data. By applying multiple machine learning classification algorithms, the system identifies high-risk applicants, helping financial institutions make data-driven lending decisions and reduce credit risk.

The project compares several classification models and selects the best-performing one based on evaluation metrics such as Accuracy, Precision, Recall, and F1-Score.

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn  
- **Environment:** Jupyter Notebook / Python IDE  

---

## 📂 Project Structure

```bash
Auto-Loan-Default-Prediction/
│
├── eda_analysis.py / eda_analysis.ipynb
│   # Exploratory Data Analysis, feature distributions, and correlation mapping
│
├── preprocessing.py
│   # Data cleaning, missing value handling, encoding, and feature scaling
│
├── model_training.py
│   # Training and evaluation of all machine learning models
│
├── random_forest_model.pkl
│   # Saved trained Random Forest model
│
└── README.md
    # Project documentation


##🧪 Models Evaluated

The following machine learning classification algorithms were implemented and compared:

-Logistic Regression (Baseline Model)
-Decision Tree Classifier
-Random Forest Classifier (Selected Final Model)
-XGBoost
-LightGBM
-Naive Bayes
-Support Vector Machine (SVM)