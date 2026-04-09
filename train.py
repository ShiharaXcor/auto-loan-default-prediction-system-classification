# train.py

from src.data_preprocessing import Preprocessor, load_csv
from src.models import train_test_split_df, train_logistic, evaluate_model, train_rf_random_search
import joblib
import os

# 1. Load raw dataset
raw = load_csv('data/raw/Automobile_Loan_Default.csv')

# 2. Preprocess data
pre = Preprocessor()
df = pre.fit_transform(raw)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split_df(df)

# 4. Train baseline Logistic Regression model
clf = train_logistic(X_train, y_train)

# 5. Evaluate
evaluate_model(clf, X_test, y_test)

# 6. (Optional) Train Random Forest with hyperparameter tuning
# rf_clf = train_rf_random_search(X_train, y_train)
# evaluate_model(rf_clf, X_test, y_test)

# 7. Save preprocessor + model
os.makedirs("models", exist_ok=True)
pre.save("models/preprocessor.joblib")
joblib.dump(clf, "models/loan_model.joblib")
print("✅ Preprocessor and model saved in models/ folder")
