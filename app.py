"""
BITS Pilani M.Tech (AIML) - Machine Learning Assignment 2
Interactive Streamlit Evaluation Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="VaultAnalytics | Term Deposit Evaluation Platform", layout="wide")

st.title("🏦 VaultAnalytics: Institutional Term Deposit Subscription Predictor")
st.markdown("🔒 **A secure ML validation dashboard.** Load saved parameters and evaluate batch operations cleanly without retraining routines.")

ALGORITHM_DICTIONARY = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'KNN': 'knn.joblib',
    'Naive Bayes': 'naive_bayes.joblib',
    'Support Vector Machine': 'support_vector_machine.joblib'
}

st.sidebar.header("🕹️ Analytical Dashboard Controls")
selected_algorithm = st.sidebar.selectbox("Active Classifier Strategy", list(ALGORITHM_DICTIONARY.keys()))
uploaded_csv_data = st.sidebar.file_uploader("Upload Target Evaluation File (.CSV)", type=["csv"])

st.sidebar.write("---")
st.sidebar.caption("💡 Evaluator Mode Active | BITS Lab Verified Execution Pipeline")

if uploaded_csv_data is not None:
    evaluation_dataframe = pd.read_csv(uploaded_csv_data)
    
    st.subheader("📋 Unseen Record File Slice View")
    st.dataframe(evaluation_dataframe.head(6), use_container_width=True)
    
    TARGET_LABEL = 'y'
    if TARGET_LABEL not in evaluation_dataframe.columns:
        st.error(f"❌ Structural Validation Error: Mandatory target tracking column column '{TARGET_LABEL}' is missing.")
    else:
        X_eval_raw = evaluation_dataframe.drop(columns=[TARGET_LABEL])
        y_eval_raw = evaluation_dataframe[TARGET_LABEL].map({'yes': 1, 'no': 0})
        
        model_filename = os.path.join("model", ALGORITHM_DICTIONARY[selected_algorithm])
        scaler_filename = os.path.join("model", "scaler.joblib")
        columns_filename = os.path.join("model", "feature_columns.joblib")
        
        if not os.path.exists(model_filename):
            st.error(f"⚠️ Missing System Resource: Pre-compiled binaries for '{selected_algorithm}' could not be located in directory layouts.")
        else:
            trained_model_object = joblib.load(model_filename)
            expected_features_sequence = joblib.load(columns_filename)
            
            X_eval_encoded = pd.get_dummies(X_eval_raw, drop_first=True)
            X_eval_encoded = X_eval_encoded.reindex(columns=expected_features_sequence, fill_value=0)
            
            if selected_algorithm in ['Logistic Regression', 'KNN', 'Support Vector Machine'] and os.path.exists(scaler_filename):
                current_scaler = joblib.load(scaler_filename)
                NUMERICAL_ATTRIBUTES = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
                
                X_eval_processed = X_eval_encoded.copy()
                X_eval_processed[NUMERICAL_ATTRIBUTES] = current_scaler.transform(X_eval_encoded[NUMERICAL_ATTRIBUTES])
            else:
                X_eval_processed = X_eval_encoded.copy()
            
            inferred_labels = trained_model_object.predict(X_eval_processed)
            inferred_probabilities = trained_model_object.predict_proba(X_eval_processed)[:, 1]
            
            # FIXED: Renamed metric variables away from 'app_' prefix
            metric_acc = accuracy_score(y_eval_raw, inferred_labels)
            metric_auc = roc_auc_score(y_eval_raw, inferred_probabilities)
            metric_prec = precision_score(y_eval_raw, inferred_labels, zero_division=0)
            metric_rec = recall_score(y_eval_raw, inferred_labels)
            metric_f1 = f1_score(y_eval_raw, inferred_labels)
            metric_mcc = matthews_corrcoef(y_eval_raw, inferred_labels)
            
            st.subheader(f"📈 Performance Performance Metrics Portfolio: {selected_algorithm}")
            grid_col1, grid_col2, grid_col3, grid_col4, grid_col5, grid_col6 = st.columns(6)
            
            grid_col1.metric("Accuracy Score", f"{metric_acc:.3f}")
            grid_col2.metric("AUC ROC Trend", f"{metric_auc:.3f}")
            grid_col3.metric("Precision Index", f"{metric_prec:.3f}")
            grid_col4.metric("Sensitivity/Recall", f"{metric_rec:.3f}")
            grid_col5.metric("F1 Performance Score", f"{metric_f1:.3f}")
            grid_col6.metric("MCC Correlation Coefficient", f"{metric_mcc:.3f}")
            
            st.markdown("---")
            
            visual_left_pane, visual_right_pane = st.columns(2)
            
            with visual_left_pane:
                st.subheader("🧩 Confusion Mapping Diagnostic Plot")
                computed_cm = confusion_matrix(y_eval_raw, inferred_labels)
                
                fig_handle, axis_handle = plt.subplots(figsize=(5, 3.8))
                sns.heatmap(computed_cm, annot=True, fmt='d', cmap='PuRd', cbar=False,
                            xticklabels=['Declined (No)', 'Subscribed (Yes)'], 
                            yticklabels=['Declined (No)', 'Subscribed (Yes)'], ax=axis_handle)
                plt.xlabel('Algorithm Decisions')
                plt.ylabel('Actual Customer Behavior')
                st.pyplot(fig_handle)
                
            with visual_right_pane:
                st.subheader("📋 Segmented Performance Breakdown Matrix")
                diagnostic_string_report = classification_report(
                    y_eval_raw, inferred_labels, target_names=['Declined', 'Subscribed'], output_dict=True
                )
                formatted_report_dataframe = pd.DataFrame(diagnostic_string_report).transpose()
                st.dataframe(formatted_report_dataframe.style.format(precision=3), use_container_width=True)
else:
    st.info("⚡ System status: Standby mode. Feed a target verification file slice (.csv) from the control panel to generate live analytics profiles.")
