import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

def extract_features(model_config):
    features = []
    for layer in model_config['layers']:
        layer_type = layer['class_name']
        layer_config = layer['config']
        features.append({
            'Layer_Type': layer_type,
            'Config': layer_config
        })
    return features

def create_feature_dataframe(features):
    flattened_data = []
    for feature in features:
        layer_data = {**{'Layer_Type': feature['Layer_Type']}, **feature['Config']}
        flattened_data.append(layer_data)
    return pd.DataFrame(flattened_data)

model_directory = "D:\\ICSE_Dataset\\H5AllBuggy"
dest_directory = "D:\\ICSE_Dataset\\Derived_Features\\code_structure_feature_buggy"
if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)

all_models_features_df = pd.DataFrame()
for model_file in os.listdir(model_directory):
    if model_file.endswith('.h5'):
        model_path = os.path.join(model_directory, model_file)
        model = tf.keras.models.load_model(model_path)
        try:
            model_config = model.get_config()
            features = extract_features(model_config)
            ff_df = create_feature_dataframe(features)
            
            numerical_columns = ff_df.select_dtypes(include=['float64', 'int64']).columns
            ff_df[numerical_columns] = ff_df[numerical_columns].fillna(0)
            scaler = StandardScaler()
            ff_df[numerical_columns] = scaler.fit_transform(ff_df[numerical_columns])
            ff_df['Sequential_Index'] = range(len(ff_df))
            ff_df['Previous_Layer_Type'] = ff_df['Layer_Type'].shift(1).fillna('None')
            ff_df['Next_Layer_Type'] = ff_df['Layer_Type'].shift(-1).fillna('None')
            desired_features = ['Layer_Type', 'Sequential_Index', 'Previous_Layer_Type', 'Next_Layer_Type']
            for column in ['activation', 'use_bias', 'trainable']:
                if column in ff_df.columns:
                    desired_features.append(column)
            ff_df_desired = ff_df[desired_features]

            layer_info = []
            for layer in model.layers:
                layer_type = type(layer).__name__
                input_shape = layer.input_shape if hasattr(layer, 'input_shape') else None
                output_shape = layer.output_shape if hasattr(layer, 'output_shape') else None
                is_reshaping_layer = layer_type in ['Flatten', 'Reshape']
                weights = layer.get_weights()
                weights_shape = weights[0].shape if weights else None
                biases_shape = weights[1].shape if len(weights) > 1 else None

                layer_info.append({
                    'input_shape': input_shape,
                    'output_shape': output_shape,
                    'is_reshaping_layer': is_reshaping_layer,
                    'weights_shape': weights_shape,
                    'biases_shape': biases_shape
                })

            layer_info_df = pd.DataFrame(layer_info)
            layer_info_df = layer_info_df.fillna('None')
            final_features_df = pd.concat([ff_df_desired, layer_info_df], axis=1)
            output_json_path = os.path.join(dest_directory, model_file.replace('.h5', '_code_structure_features.json'))
            final_features_df.to_json(output_json_path, orient='records', lines=True)
            all_models_features_df = pd.concat([all_models_features_df, final_features_df], ignore_index=True)
            
        except Exception as e:
            print(f"Error loading model {model_file}: {e}")