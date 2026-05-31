import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import random
from xgboost import XGBClassifier
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import shap
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV

# Batch Size 
data = pd.read_csv("D..\\hyperparameter_3rd_level.csv")
df_combined = data.copy()
df_combined['label'] = df_combined['label'].apply(lambda x: 1 if x == 'HBS' else 0)
X = df_combined.drop(columns=['label'])
y = df_combined['label']
X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
X = X.replace([np.inf, -np.inf], np.nan).apply(lambda x: x.fillna(x.max()), axis=0)

for column in X.columns:
    if X[column].isna().sum() > 0 or np.isinf(X[column]).sum() > 0:
        X[column] = X[column].replace([np.inf, -np.inf], np.nan).fillna(X[column].mean())
    lower_quantile = X[column].quantile(0.01)
    upper_quantile = X[column].quantile(0.99)
    X[column] = X[column].clip(lower=lower_quantile, upper=upper_quantile)

X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
X = X.replace([np.inf, -np.inf], np.nan).apply(lambda x: x.fillna(x.mean()), axis=0)

if X.isna().sum().sum() > 0 or np.isinf(X).sum().sum() > 0:
    print("There are still NaNs or infinite values in the DataFrame. Additional cleaning needed.")
else:
    print("Data cleaning completed successfully. No NaNs or infinite values remain.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
def run_models_classification_report(X_train, X_test, y_train, y_test, verbose=True):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'SVC': SVC(max_iter=1000),
        'KNeighbors': KNeighborsClassifier(),
        'MLP': MLPClassifier(max_iter=1000),
        'GradientBoosting': GradientBoostingClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'Bagging': BaggingClassifier(),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    cv_results = {}

    for model_name, model in tqdm(models.items(), desc='Training Models', total=len(models)):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = classification_report(y_test, y_pred, output_dict=True)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_results[model_name] = cv_scores
        
        if verbose:
            print(model_name)
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            print("Cross-Validation Scores:")
            print(cv_scores)
            print("Mean CV Score:", cv_scores.mean())
            print("Standard Deviation of CV Scores:", cv_scores.std())
            print()
            
    return results, cv_results

results, cv_results = run_models_classification_report(X_train, X_test, y_train, y_test, verbose=False)
for model_name in results.keys():
    print(model_name)
    print("Classification Report:")
    print(results[model_name])
    print("Cross-Validation Scores:")
    print(cv_results[model_name])
    print("Mean CV Score:", cv_results[model_name].mean())
    print("Standard Deviation of CV Scores:", cv_results[model_name].std())
    print()


# Learning Rate
data = pd.read_csv("..\\hyperparameter_3rd_level.csv")
df_combined = data.copy()
df_combined['label'] = df_combined['label'].apply(lambda x: 1 if x == 'HLR' else 0)

X = df_combined.drop(columns=['label'])
y = df_combined['label']
X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
X = X.replace([np.inf, -np.inf], np.nan).apply(lambda x: x.fillna(x.max()), axis=0)

for column in X.columns:
    if X[column].isna().sum() > 0 or np.isinf(X[column]).sum() > 0:
        X[column] = X[column].replace([np.inf, -np.inf], np.nan).fillna(X[column].mean())
    lower_quantile = X[column].quantile(0.01)
    upper_quantile = X[column].quantile(0.99)
    X[column] = X[column].clip(lower=lower_quantile, upper=upper_quantile)

X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
X = X.replace([np.inf, -np.inf], np.nan).apply(lambda x: x.fillna(x.mean()), axis=0)

if X.isna().sum().sum() > 0 or np.isinf(X).sum().sum() > 0:
    print("There are still NaNs or infinite values in the DataFrame. Additional cleaning needed.")
else:
    print("Data cleaning completed successfully. No NaNs or infinite values remain.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
def run_models_classification_report(X_train, X_test, y_train, y_test, verbose=True):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'SVC': SVC(max_iter=1000),
        'KNeighbors': KNeighborsClassifier(),
        'MLP': MLPClassifier(max_iter=1000),
        'GradientBoosting': GradientBoostingClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'Bagging': BaggingClassifier(),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    cv_results = {}

    for model_name, model in tqdm(models.items(), desc='Training Models', total=len(models)):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = classification_report(y_test, y_pred, output_dict=True)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_results[model_name] = cv_scores
        
        if verbose:
            print(model_name)
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            print("Cross-Validation Scores:")
            print(cv_scores)
            print("Mean CV Score:", cv_scores.mean())
            print("Standard Deviation of CV Scores:", cv_scores.std())
            print()
            
    return results, cv_results


results, cv_results = run_models_classification_report(X_train, X_test, y_train, y_test, verbose=False)
for model_name in results.keys():
    print(model_name)
    print("Classification Report:")
    print(results[model_name])
    print("Cross-Validation Scores:")
    print(cv_results[model_name])
    print("Mean CV Score:", cv_results[model_name].mean())
    print("Standard Deviation of CV Scores:", cv_results[model_name].std())
    print()

# number of epoch
data = pd.read_csv("..\\hyperparameter_3rd_level.csv")
df_combined = data.copy()
df_combined['label'] = df_combined['label'].apply(lambda x: 1 if x == 'HNE' else 0)

X = df_combined.drop(columns=['label'])
y = df_combined['label']
X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
X = X.replace([np.inf, -np.inf], np.nan).apply(lambda x: x.fillna(x.max()), axis=0)

for column in X.columns:
    if X[column].isna().sum() > 0 or np.isinf(X[column]).sum() > 0:
        X[column] = X[column].replace([np.inf, -np.inf], np.nan).fillna(X[column].mean())
    lower_quantile = X[column].quantile(0.01)
    upper_quantile = X[column].quantile(0.99)
    X[column] = X[column].clip(lower=lower_quantile, upper=upper_quantile)

X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
X = X.replace([np.inf, -np.inf], np.nan).apply(lambda x: x.fillna(x.mean()), axis=0)

if X.isna().sum().sum() > 0 or np.isinf(X).sum().sum() > 0:
    print("There are still NaNs or infinite values in the DataFrame. Additional cleaning needed.")
else:
    print("Data cleaning completed successfully. No NaNs or infinite values remain.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
def run_models_classification_report(X_train, X_test, y_train, y_test, verbose=True):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'SVC': SVC(max_iter=1000),
        'KNeighbors': KNeighborsClassifier(),
        'MLP': MLPClassifier(max_iter=1000),
        'GradientBoosting': GradientBoostingClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'Bagging': BaggingClassifier(),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    cv_results = {}

    for model_name, model in tqdm(models.items(), desc='Training Models', total=len(models)):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = classification_report(y_test, y_pred, output_dict=True)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_results[model_name] = cv_scores
        
        if verbose:
            print(model_name)
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            print("Cross-Validation Scores:")
            print(cv_scores)
            print("Mean CV Score:", cv_scores.mean())
            print("Standard Deviation of CV Scores:", cv_scores.std())
            print()
            
    return results, cv_results


results, cv_results = run_models_classification_report(X_train, X_test, y_train, y_test, verbose=False)
for model_name in results.keys():
    print(model_name)
    print("Classification Report:")
    print(results[model_name])
    print("Cross-Validation Scores:")
    print(cv_results[model_name])
    print("Mean CV Score:", cv_results[model_name].mean())
    print("Standard Deviation of CV Scores:", cv_results[model_name].std())
    print()


# disable batching
data = pd.read_csv("..\\hyperparameter_3rd_level.csv")
df_combined = data.copy()
df_combined['label'] = df_combined['label'].apply(lambda x: 1 if x == 'HBE' else 0)

X = df_combined.drop(columns=['label'])
y = df_combined['label']
X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
X = X.replace([np.inf, -np.inf], np.nan).apply(lambda x: x.fillna(x.max()), axis=0)

for column in X.columns:
    if X[column].isna().sum() > 0 or np.isinf(X[column]).sum() > 0:
        X[column] = X[column].replace([np.inf, -np.inf], np.nan).fillna(X[column].mean())
    lower_quantile = X[column].quantile(0.01)
    upper_quantile = X[column].quantile(0.99)
    X[column] = X[column].clip(lower=lower_quantile, upper=upper_quantile)

X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
X = X.replace([np.inf, -np.inf], np.nan).apply(lambda x: x.fillna(x.mean()), axis=0)

if X.isna().sum().sum() > 0 or np.isinf(X).sum().sum() > 0:
    print("There are still NaNs or infinite values in the DataFrame. Additional cleaning needed.")
else:
    print("Data cleaning completed successfully. No NaNs or infinite values remain.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
def run_models_classification_report(X_train, X_test, y_train, y_test, verbose=True):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'SVC': SVC(max_iter=1000),
        'KNeighbors': KNeighborsClassifier(),
        'MLP': MLPClassifier(max_iter=1000),
        'GradientBoosting': GradientBoostingClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'Bagging': BaggingClassifier(),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    cv_results = {}

    for model_name, model in tqdm(models.items(), desc='Training Models', total=len(models)):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = classification_report(y_test, y_pred, output_dict=True)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_results[model_name] = cv_scores
        
        if verbose:
            print(model_name)
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            print("Cross-Validation Scores:")
            print(cv_scores)
            print("Mean CV Score:", cv_scores.mean())
            print("Standard Deviation of CV Scores:", cv_scores.std())
            print()
            
    return results, cv_results


results, cv_results = run_models_classification_report(X_train, X_test, y_train, y_test, verbose=False)
for model_name in results.keys():
    print(model_name)
    print("Classification Report:")
    print(results[model_name])
    print("Cross-Validation Scores:")
    print(cv_results[model_name])
    print("Mean CV Score:", cv_results[model_name].mean())
    print("Standard Deviation of CV Scores:", cv_results[model_name].std())
    print()