"""
BITS Pilani M.Tech (AIML) - Machine Learning Assignment 2
Main Evaluation and Training Pipeline
Dataset: UCI Bank Marketing (Term Deposit Subscription Prediction)
"""

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
import joblib

# Setup isolated directory map
os.makedirs("model", exist_ok=True)

# 1. Fetch data directly from authentic UCI repository source
DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
ZIP_PATH = "bank_marketing.zip"

print("📡 Connecting to UCI Repository to pull Bank Marketing Dataset...")
try:
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall("bank_extracted")
    with zipfile.ZipFile("bank_extracted/bank.zip", 'r') as bank_data_zip:
        with bank_data_zip.open("bank-full.csv") as bank_csv:
            raw_bank_df = pd.read_csv(bank_csv, sep=';')
    print(f"✅ Target dataset loaded successfully. Shape: {raw_bank_df.shape}")
except Exception as e:
    print(f"❌ Error downloading data: {e}")
    print("💡 Fallback: Generating synthetically structured bank dataset for lab preview...")
    np.random.seed(101)
    samples = 1000
    mock_data = {
        'age': np.random.randint(18, 70, samples),
        'balance': np.random.randint(-500, 5000, samples),
        'day': np.random.randint(1, 31, samples),
        'duration': np.random.randint(10, 1500, samples),
        'campaign': np.random.randint(1, 10, samples),
        'pdays': np.random.choice([-1, 999, 10, 20], samples),
        'previous': np.random.randint(0, 5, samples),
        'job': np.random.choice(['admin.', 'technician', 'services', 'management'], samples),
        'marital': np.random.choice(['married', 'single', 'divorced'], samples),
        'education': np.random.choice(['primary', 'secondary', 'tertiary'], samples),
        'default': np.random.choice(['no', 'yes'], samples, p=[0.98, 0.02]),
        'housing': np.random.choice(['no', 'yes'], samples),
        'loan': np.random.choice(['no', 'yes'], samples),
        'contact': np.random.choice(['cellular', 'telephone'], samples),
        'month': np.random.choice(['may', 'jun', 'jul', 'aug'], samples),
        'poutcome': np.random.choice(['unknown', 'failure', 'success'], samples),
        'y': np.random.choice(['no', 'yes'], samples, p=[0.88, 0.12])
    }
    raw_bank_df = pd.DataFrame(mock_data)

# Convert target 'yes'/'no' into numeric binary flags
raw_bank_df['y'] = raw_bank_df['y'].map({'yes': 1, 'no': 0})

features_df = raw_bank_df.drop(columns=['y'])
target_vector = raw_bank_df['y']

processed_features_df = pd.get_dummies(features_df, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    processed_features_df, target_vector, test_size=0.25, random_state=101, stratify=target_vector
)

numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
banking_scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_cols] = banking_scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = banking_scaler.transform(X_test[numerical_cols])

joblib.dump(banking_scaler, 'model/scaler.joblib')
joblib.dump(processed_features_df.columns.tolist(), 'model/feature_columns.joblib')

streamlit_test_set = X_test.copy()
streamlit_test_set['y'] = y_test.map({1: 'yes', 0: 'no'})
streamlit_test_set.to_csv('test_data.csv', index=False)
print("💾 Extracted standalone 'test_data.csv' for web application analysis.")

engine_classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=101),
    'Decision Tree': DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=101),
    'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance'),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(probability=True, kernel='rbf', max_iter=5000, random_state=101)
}

evaluation_records = []
print("\n🏋️ Training classifiers and storing individual binary artifacts...")
for model_key, algorithm in engine_classifiers.items():
    if model_key in ['Logistic Regression', 'KNN', 'Support Vector Machine']:
        algorithm.fit(X_train_scaled, y_train)
        predictions = algorithm.predict(X_test_scaled)
        probabilities = algorithm.predict_proba(X_test_scaled)[:, 1]
    else:
        algorithm.fit(X_train, y_train)
        predictions = algorithm.predict(X_test)
        probabilities = algorithm.predict_proba(X_test)[:, 1]
        
    safe_filename = f"model/{model_key.lower().replace(' ', '_')}.joblib"
    joblib.dump(algorithm, safe_filename)
    print(f"   -> Successfully saved: {safe_filename}")
    
    acc_metric = accuracy_score(y_test, predictions)
    auc_metric = roc_auc_score(y_test, probabilities)
    prec_metric = precision_score(y_test, predictions, zero_division=0)
    rec_metric = recall_score(y_test, predictions)
    f1_metric = f1_score(y_test, predictions)
    mcc_metric = matthews_corrcoef(y_test, predictions)
    
    evaluation_records.append({
        'ML Model Name': model_key,
        'Accuracy': round(acc_metric, 3),
        'AUC': round(auc_metric, 3),
        'Precision': round(prec_metric, 3),
        'Recall': round(rec_metric, 3),
        'F1': round(f1_metric, 3),
        'MCC': round(mcc_metric, 3)
    })

summary_metrics_table = pd.DataFrame(evaluation_records)
print("\n📊 --- COPY THIS DATA TABLE INTO YOUR README.MD FILE ---")
print(summary_metrics_table.to_markdown(index=False))

if os.path.exists(ZIP_PATH):
    os.remove(ZIP_PATH)
