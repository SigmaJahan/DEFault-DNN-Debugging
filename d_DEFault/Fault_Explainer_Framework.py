import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import shap
import xgboost as xgb
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay


df = pd.read_csv('layer_activation_df.csv')
target_column = 'Buggy'
X = df.drop(columns=[target_column])
y = df[target_column]
X = X.drop(columns=['Model_File'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
random_instance = random.randint(0, len(X_test))
sample = X_test.iloc[random_instance]
sample_df = pd.DataFrame([sample])

model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample_df)
feature_names = X.columns
shap.summary_plot(shap_values, sample_df, feature_names=feature_names)
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=feature_names, class_names=['Correct', 'Buggy'], discretize_continuous=True)
i = 0
exp = explainer.explain_instance(X_test.values[i], model.predict_proba, num_features=5)
exp.show_in_notebook(show_table=True, show_all=False)

fault_messages = {
    'CountDense': 'Check the configuration and number of Dense layers.',
    'CountActivation': 'Look into the types and placements of activation functions.',
    'CountDropout': 'Look into the dropout rate and the layers that use dropout.',
    'CountFlatten': 'Verify the use of Flatten layers, especially before Dense layers.',
    'CountInputLayer': 'Check the configuration of the input layer.',
    'CountConv1D': 'Inspect the configuration of 1D convolutional layers.',
    'CountConv2D': 'Inspect the configuration of 2D convolutional layers.',
    'CountConv3D': 'Inspect the configuration of 3D convolutional layers.',
    'CountMaxPooling1D': 'Verify the use of 1D max pooling layers.',
    'CountMaxPooling2D': 'Verify the use of 2D max pooling layers.',
    'CountMaxPooling3D': 'Verify the use of 3D max pooling layers.',
    'CountSimpleRNN': 'Check the configuration of SimpleRNN layers.',
    'CountGRU': 'Check the configuration of GRU layers.',
    'CountLSTM': 'Check the configuration of LSTM layers.',
    'CountEmbedding': 'Verify the configuration of embedding layers, especially in NLP tasks.',
    'CountBatchNormalization': 'Ensure the BatchNormalization layers are correctly placed.',
    'CountLayerNormalization': 'Ensure the LayerNormalization layers are correctly placed.',
    'Countsoftmax': 'Look into the activation function Softmax and its placement.',
    'Countrelu': 'Look into the activation function ReLU and its placement.',
    'Counttanh': 'Look into the activation function Tanh and its placement.',
    'Countsigmoid': 'Look into the activation function Sigmoid and layers that use Sigmoid.',
    'Counthard_sigmoid': 'Look into the activation function Hard Sigmoid and its placement.',
    'Countexponential': 'Look into the activation function Exponential and its placement.',
    'Countlinear': 'Look into the activation function Linear and its placement.',
    'CountL1': 'Inspect the usage of L1 regularization.',
    'CountL2': 'Inspect the usage of L2 regularization.',
    'CountL1L2': 'Inspect the usage of combined L1 and L2 regularization.',
    'Total_Params': 'Inspect the total number of parameters for potential issues.',
    'Trainable_Params': 'Check the number of trainable parameters, which might be too high or too low.',
    'Pooling_Amounts': 'Verify the pooling amounts in pooling layers.',
    'Filter_Sizes': 'Inspect the filter sizes in convolutional layers.',
    'Paddings': 'Verify the padding settings in convolutional layers.',
    'Strides': 'Check the stride values in convolutional layers.',
    'Num_Neurons': 'Inspect the number of neurons in Dense layers.',
    'Output_Shapes': 'Verify the output shapes of different layers.',
    'Dropout_Rates': 'Look into the dropout rate and the layers that use dropout.',
    'Number_of_Layers': 'Check the depth and placement of layers within the network.',
    'Max_Neurons': 'Verify the maximum number of neurons in any single layer.',
    'Activation_Presence': 'Ensure appropriate activation functions are present in the network.',
    'Max_Output_Shape': 'Verify the maximum output shape for layers.'
    # Can add more mappings as needed
}

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

explainer = shap.TreeExplainer(model)
random_instance = np.random.randint(0, len(X_test))
sample = X_test.iloc[random_instance]
sample_df = pd.DataFrame([sample])
shap_values = explainer.shap_values(sample_df)
feature_names = X.columns
shap.summary_plot(shap_values, sample_df, feature_names=feature_names)

explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=feature_names, class_names=['Correct', 'Buggy'], discretize_continuous=True)
i = 0
exp = explainer.explain_instance(X_test.values[i], model.predict_proba, num_features=5)
exp.show_in_notebook(show_table=True, show_all=False)

features = [0, 1, 2]  
PartialDependenceDisplay.from_estimator(clf, X_train, features, feature_names=X.columns, grid_resolution=10)

explainer_shap = shap.TreeExplainer(clf)
random_instance = np.random.randint(0, len(X_test))
sample = X_test.iloc[0]
sample_df = pd.DataFrame([sample])
shap_values = explainer_shap.shap_values(sample_df)
feature_names = X.columns

if isinstance(shap_values, list):
    shap_values = shap_values[1] 
shap_values = np.array(shap_values).reshape(1, -1) 
shap_values = shap_values[:, :len(X.columns)]

print(f"Shape of SHAP values: {shap_values.shape}")
print(f"Shape of X columns: {X.columns.shape}")
shap_important_features = pd.DataFrame(shap_values, columns=X.columns).abs().mean().sort_values(ascending=False).head(5).index.tolist()

def print_fault_messages(important_features):
    for feature in important_features:
        if feature in fault_messages:
            print(f"Feature: {feature}, Insight: {fault_messages[feature]}")

print("Faults based on SHAP important features:")
print_fault_messages(shap_important_features)


