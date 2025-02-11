import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load
import configparser
from pathlib import Path

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
sys.path.append(BASE_DIR)

def load_classifier(filename):
    full_path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(full_path):
        return load(full_path)
    else:
        raise FileNotFoundError(f"Model file {full_path} not found. Please ensure the model is saved in the specified directory.")

try:
    best_clf_detection = load_classifier('best_clf_detection.joblib')
    best_clf_activation = load_classifier('best_clf_activation.joblib')
    best_clf_layer = load_classifier('best_clf_layer.joblib')
    best_clf_hyperparameter = load_classifier('best_clf_hyperparameter.joblib')
    best_clf_loss = load_classifier('best_clf_loss.joblib')
    best_clf_optimization = load_classifier('best_clf_optimization.joblib')
    best_clf_regularizer = load_classifier('best_clf_regularizer.joblib')
    best_clf_weights = load_classifier('best_clf_weights.joblib')
except FileNotFoundError as e:
    print(str(e))
    sys.exit(1)  

classifiers = [
    (best_clf_detection, 'Detection'),
    (best_clf_activation, 'Activation'),
    (best_clf_layer, 'Layer'),
    (best_clf_hyperparameter, 'Hyperparameter'),
    (best_clf_loss, 'Loss'),
    (best_clf_optimization, 'Optimization'),
    (best_clf_regularizer, 'Regularizer'),
    (best_clf_weights, 'Weights')
]

columns_to_keep = [
    'gpu_memory_utilization', 'cpu_utilization', 'train_acc', 'val_acc', 'memory_usage',
    'loss_oscillation', 'acc_gap_too_big', 'adjusted_lr', 'dying_relu', 'gradient_vanish',
    'gradient_explode', 'gradient_median', 'std_grad', 'gradient_min', 'mean_grad',
    'large_weight_count', 'mean_activation', 'std_activation', 'gradient_std', 'gradient_max'
]

def convert_tensor_to_bool(tensor_str):
    if isinstance(tensor_str, str):
        return 'True' in tensor_str
    return bool(tensor_str)

import os
import configparser

def load_classifier_thresholds(filepath="config/config.ini"):
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    config_path = os.path.join(script_dir, filepath)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    config = configparser.ConfigParser()
    config.clear()
    config.read(config_path)

    if 'ClassifierThresholds' in config:
        thresholds = {k.lower(): float(v) for k, v in config['ClassifierThresholds'].items()}
        return thresholds
    else:
        raise KeyError("The 'ClassifierThresholds' section was not found in the config file.")

        
classifier_thresholds = load_classifier_thresholds()

def preprocess_data(df):
    if 'dying_relu' in df.columns:
        df['dying_relu'] = df['dying_relu'].apply(convert_tensor_to_bool).astype(int)
    if 'saturated_activation' in df.columns:
        df['saturated_activation'] = df['saturated_activation'].apply(convert_tensor_to_bool).astype(int)

    df = df[columns_to_keep].copy()
    df.fillna(df.mean(), inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    for column in df.columns:
        lower_quantile = df[column].quantile(0.01)
        upper_quantile = df[column].quantile(0.99)
        df[column] = df[column].clip(lower=lower_quantile, upper=upper_quantile)
    for col in columns_to_keep:
        if col not in df.columns:
            df[col] = 0  
    df = df[columns_to_keep]
    return df

def process_test_files(path, classifiers):
    results = []
    scaler = StandardScaler() 

    files_to_process = []
    if os.path.isfile(path):
        files_to_process = [path]
    elif os.path.isdir(path):
        files_to_process = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    else:
        raise FileNotFoundError(f"The path {path} does not exist or is not a file/directory.")

    for clf, clf_name in classifiers:
        classifier_results = []
        probabilities = []
        for file_path in files_to_process:
            df = pd.read_csv(file_path)
            if df.empty:
                continue

            df_processed = preprocess_data(df)
            if df_processed.isna().sum().sum() > 0:
                df_processed.fillna(0, inplace=True)

            X = scaler.fit_transform(df_processed)
            if np.isnan(X).any() or np.isinf(X).any():
                X = np.nan_to_num(X)

            proba = clf.predict_proba(X)
            positive_class_probs = [p[1] for p in proba]
            avg_probability = np.mean(positive_class_probs)
            probabilities.append(avg_probability)

        threshold = classifier_thresholds.get(clf_name.lower(), 0.5)

        for file_path in files_to_process:
            df = pd.read_csv(file_path)
            if df.empty:
                continue

            df_processed = preprocess_data(df)
            if df_processed.isna().sum().sum() > 0:
                df_processed.fillna(0, inplace=True)

            X = scaler.transform(df_processed)
            if np.isnan(X).any() or np.isinf(X).any():
                X = np.nan_to_num(X)

            proba = clf.predict_proba(X)
            positive_class_probs = [p[1] for p in proba]
            avg_probability = np.mean(positive_class_probs)
            overall_label = 'Positive' if avg_probability >= threshold else 'Negative'

            classifier_results.append({
                'filename': os.path.basename(file_path),
                'average_probability': avg_probability,
                'overall_label': overall_label,
                'classifier': clf_name
            })

        results.extend(classifier_results)

    return pd.DataFrame(results)

File_Path_Test_File = os.path.join(BASE_DIR, "data", "pixelcnn_buggy.csv")
df_results = process_test_files(File_Path_Test_File, classifiers)

def generate_user_friendly_output(df_results):
    df_results.rename(columns={
        'filename': 'File Name',
        'average_probability': 'Avg. Probability (%)',
        'overall_label': 'Bug Detected?',
        'classifier': 'Bug Category'
    }, inplace=True)


    df_results['Avg. Probability (%)'] = (df_results['Avg. Probability (%)'] * 100).round(1)
    df_results['Bug Detected?'] = df_results['Bug Detected?'].apply(lambda x: 'Yes ✅' if x == 'Positive' else 'No ❌')

    print("\n=== Detailed Results ===")
    print(df_results.to_string(index=False))

    df_pivot = df_results.pivot_table(index='File Name', columns='Bug Category', values='Bug Detected?', aggfunc='first')
    df_pivot.fillna("No ❌", inplace=True)

generate_user_friendly_output(df_results)