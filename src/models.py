import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_test_split_df(df: pd.DataFrame, target_col='Default', test_size=0.2, random_state=42):
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    return train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)


def train_logistic(X_train, y_train, class_weight='balanced'):
    clf = LogisticRegression(max_iter=1000, class_weight=class_weight, solver='liblinear')
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    print("PR AUC:", auc(recall, precision))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return {'y_pred': y_pred, 'y_proba': y_proba}


def train_rf_random_search(X_train, y_train, n_iter=10, n_jobs=-1, random_state=42):
    rf = RandomForestClassifier(n_jobs=n_jobs, class_weight='balanced', random_state=random_state)
    param_dist = {
        'n_estimators': [100, 200, 400],
        'max_depth': [6, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    rs = RandomizedSearchCV(rf, param_dist, n_iter=n_iter, scoring='roc_auc', cv=cv, n_jobs=n_jobs, random_state=random_state, verbose=1)
    rs.fit(X_train, y_train)
    return rs.best_estimator_, rs.best_params_


def save_model(clf, path: str):
    joblib.dump(clf, path)


def load_model(path: str):
    return joblib.load(path)
