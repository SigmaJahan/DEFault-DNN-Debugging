import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from Fault_Dynamic_Activation import best_clf as best_clf_activation
from Fault_Dynamic_Layer import best_clf as best_clf_layer
from Fault_Dynamic_Hyperparameter import best_clf as best_clf_hyperparameter
from Fault_Dynamic_Loss import best_clf as best_clf_loss
from Fault_Dynamic_Optimization import best_clf as best_clf_optimization

def convert_tensor_to_bool(tensor_str):
    if isinstance(tensor_str, str):
        return 'True' in tensor_str
    elif isinstance(tensor_str, bool):
        return tensor_str
    elif isinstance(tensor_str, int):
        return bool(tensor_str)
    return False

# List of columns to keep based on feature importance
columns_to_keep = [
    'gpu_memory_utilization', 'cpu_utilization', 'train_acc', 'val_acc', 'memory_usage',
    'loss_oscillation', 'acc_gap_too_big', 'adjusted_lr', 'dying_relu', 'gradient_vanish',
    'gradient_explode', 'gradient_median', 'std_grad', 'gradient_min', 'mean_grad',
    'large_weight_count', 'mean_activation', 'std_activation', 'gradient_std', 'gradient_max'
]

def process_test_files(directory, columns_to_keep, classifier, classifier_name):
    results = []
    probabilities = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df_test = pd.read_csv(file_path)
            if len(df_test) == 0:
                continue
            if 'dying_relu' in df_test.columns:
                df_test['dying_relu'] = df_test['dying_relu'].apply(lambda x: convert_tensor_to_bool(x))
                df_test['dying_relu'] = df_test['dying_relu'].apply(lambda x: 1 if x else 0)
            if 'saturated_activation' in df_test.columns:
                df_test['saturated_activation'] = df_test['saturated_activation'].apply(lambda x: 1 if x else 0)

            # Keep only the top 20 features
            df_filtered = df_test[columns_to_keep]
            X = df_filtered.copy()
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
                print(f"There are still NaNs or infinite values in the DataFrame for file {filename}. Additional cleaning needed.")
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            y_new_pred_proba = classifier.predict_proba(X_scaled)
            positive_class_probabilities = [proba[1] for proba in y_new_pred_proba]
            average_probability = sum(positive_class_probabilities) / len(positive_class_probabilities)
            probabilities.append(average_probability)

    gmm = GaussianMixture(n_components=2, random_state=0).fit(np.array(probabilities).reshape(-1, 1))
    means = gmm.means_.flatten()
    threshold = np.mean(means) 

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df_test = pd.read_csv(file_path)
            if len(df_test) == 0:
                continue
            if 'dying_relu' in df_test.columns:
                df_test['dying_relu'] = df_test['dying_relu'].apply(lambda x: convert_tensor_to_bool(x))
                df_test['dying_relu'] = df_test['dying_relu'].apply(lambda x: 1 if x else 0)
            if 'saturated_activation' in df_test.columns:
                df_test['saturated_activation'] = df_test['saturated_activation'].apply(lambda x: 1 if x else 0)

            # Keep only the top 20 features
            df_filtered = df_test[columns_to_keep]
            X = df_filtered.copy()
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
                print(f"There are still NaNs or infinite values in the DataFrame for file {filename}. Additional cleaning needed.")
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            y_new_pred_proba = classifier.predict_proba(X_scaled)
            positive_class_probabilities = [proba[1] for proba in y_new_pred_proba]
            average_probability = sum(positive_class_probabilities) / len(positive_class_probabilities)

            overall_label = 'Positive' if average_probability >= threshold else 'Negative'

            results.append({
                'filename': filename,
                'average_probability': average_probability,
                'overall_label': overall_label,
                'classifier': classifier_name
            })
    
    return results

classifiers = [
    (best_clf_activation, 'Activation'),
    (best_clf_layer, 'Layer'),
    (best_clf_hyperparameter, 'Hyperparameter'),
    (best_clf_loss, 'Loss'),
    (best_clf_optimization, 'Optimization')
]

test_file_path = "D:\\ICSE_Dataset\\test"
if not os.path.exists(test_file_path):
    raise FileNotFoundError(f"The file {test_file_path} does not exist")
all_results = []
for clf, clf_name in classifiers:
    results = process_test_files(test_file_path, columns_to_keep, clf, clf_name)
    all_results.extend(results)
for result in all_results:
    print(f"File: {result['filename']}, Classifier: {result['classifier']}, Average Probability of Positive class = {result['average_probability']:.2f}, Overall Predicted label = {result['overall_label']}")

df_results = pd.DataFrame(all_results)
df_pivot = df_results.pivot_table(index='filename', columns='classifier', values='overall_label', aggfunc='first')


import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from Fault_Detection import best_clf as best_clf_detection

def convert_tensor_to_bool(tensor_str):
    if isinstance(tensor_str, str):
        return 'True' in tensor_str
    elif isinstance(tensor_str, bool):
        return tensor_str
    elif isinstance(tensor_str, int):
        return bool(tensor_str)
    return False

# List of columns to keep based on feature importance
columns_to_keep = [
    'gpu_memory_utilization', 'cpu_utilization', 'train_acc', 'val_acc', 'memory_usage',
    'loss_oscillation', 'acc_gap_too_big', 'adjusted_lr', 'dying_relu', 'gradient_vanish',
    'gradient_explode', 'gradient_median', 'std_grad', 'gradient_min', 'mean_grad',
    'large_weight_count', 'mean_activation', 'std_activation', 'gradient_std', 'gradient_max'
]

def process_test_files(directory, columns_to_keep, classifier, classifier_name):
    results = []
    probabilities = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df_test = pd.read_csv(file_path)
            if len(df_test) == 0:
                continue
            if 'dying_relu' in df_test.columns:
                df_test['dying_relu'] = df_test['dying_relu'].apply(lambda x: convert_tensor_to_bool(x))
                df_test['dying_relu'] = df_test['dying_relu'].apply(lambda x: 1 if x else 0)
            if 'saturated_activation' in df_test.columns:
                df_test['saturated_activation'] = df_test['saturated_activation'].apply(lambda x: 1 if x else 0)

            df_filtered = df_test[columns_to_keep]
            X = df_filtered.copy()
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
                print(f"There are still NaNs or infinite values in the DataFrame for file {filename}. Additional cleaning needed.")
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            y_new_pred_proba = classifier.predict_proba(X_scaled)
            positive_class_probabilities = [proba[1] for proba in y_new_pred_proba]
            average_probability = sum(positive_class_probabilities) / len(positive_class_probabilities)
            probabilities.append(average_probability)

    gmm = GaussianMixture(n_components=2, random_state=0).fit(np.array(probabilities).reshape(-1, 1))
    means = gmm.means_.flatten()
    threshold = np.mean(means) 

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df_test = pd.read_csv(file_path)
            if len(df_test) == 0:
                continue
            if 'dying_relu' in df_test.columns:
                df_test['dying_relu'] = df_test['dying_relu'].apply(lambda x: convert_tensor_to_bool(x))
                df_test['dying_relu'] = df_test['dying_relu'].apply(lambda x: 1 if x else 0)
            if 'saturated_activation' in df_test.columns:
                df_test['saturated_activation'] = df_test['saturated_activation'].apply(lambda x: 1 if x else 0)

            # Keep only the top 20 features
            df_filtered = df_test[columns_to_keep]
            X = df_filtered.copy()
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
                print(f"There are still NaNs or infinite values in the DataFrame for file {filename}. Additional cleaning needed.")
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            y_new_pred_proba = classifier.predict_proba(X_scaled)
            positive_class_probabilities = [proba[1] for proba in y_new_pred_proba]
            average_probability = sum(positive_class_probabilities) / len(positive_class_probabilities)

            overall_label = 'Positive' if average_probability >= threshold else 'Negative'

            results.append({
                'filename': filename,
                'average_probability': average_probability,
                'overall_label': overall_label,
                'classifier': classifier_name
            })
    
    return results

classifiers = [
    (best_clf_detection, 'Detection')
]

test_file_path = "D:\\ICSE_Dataset\\test"
if not os.path.exists(test_file_path):
    raise FileNotFoundError(f"The file {test_file_path} does not exist")
all_results = []
for clf, clf_name in classifiers:
    results = process_test_files(test_file_path, columns_to_keep, clf, clf_name)
    all_results.extend(results)
for result in all_results:
    print(f"File: {result['filename']}, Classifier: {result['classifier']}, Average Probability of Positive class = {result['average_probability']:.2f}, Overall Predicted label = {result['overall_label']}")

df_evaluation = pd.read_csv("..\Evaluation.csv")


