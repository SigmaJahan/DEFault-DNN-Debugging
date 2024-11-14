# Import required libraries
import pandas as pd
import numpy as np
import random
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('..\\static_features_df.csv') 
target_column = 'Buggy'
X = df.drop(columns=[target_column])
y = df[target_column]
X = X.drop(columns=['Model_File'], errors='ignore')  #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
test_df = pd.read_csv('..\\static_features_df_test_file.csv')  
unseen_sample = test_df.drop(columns=[target_column]).iloc[0]  
unseen_sample_df = pd.DataFrame([unseen_sample], columns=X.columns)

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
}

explainer_shap = shap.TreeExplainer(model)
shap_values = explainer_shap.shap_values(unseen_sample_df)
if isinstance(shap_values, list):
    shap_values = shap_values[1]
shap_important_features = pd.DataFrame(shap_values, columns=X.columns).abs().mean().sort_values(ascending=False).head(5).index.tolist()
shap_faults = {feature: fault_messages.get(feature, "No specific fault message") for feature in shap_important_features}
print("SHAP Important Features:", shap_important_features)
print("SHAP Fault Insights:")
for feature, insight in shap_faults.items():
    print(f"{feature}: {insight}")

shap.initjs()
shap.force_plot(explainer_shap.expected_value, shap_values, unseen_sample_df)