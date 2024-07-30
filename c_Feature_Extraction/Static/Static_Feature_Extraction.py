import os
import random
import pandas as pd
import tensorflow as tf
import numpy as np

model_directory = "D:\\ICSE_Dataset\\H5AllCorrect"
dest_directory = "."
if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)

correct_files = os.listdir(model_directory)
correct_files = random.sample(correct_files, 1682)
all_models_features_df = pd.DataFrame()

keras_layers = [
    "Dense", "Activation", "Dropout", "Flatten", "InputLayer",
    "Conv1D", "Conv2D", "Conv3D",
    "MaxPooling1D", "MaxPooling2D", "MaxPooling3D",
    "SimpleRNN", "GRU", "LSTM",
    "Embedding", "BatchNormalization", "LayerNormalization"
]

keras_activations = [
    "softmax", "relu", "tanh", "sigmoid", "hard_sigmoid",
    "exponential", "linear"
]

keras_regularizations = [
    "L1", "L2", "L1L2"
]

def get_max_min_dimension(shape_list):
    max_dim = 0
    min_dim = float('inf')
    for shape in shape_list:
        for dim in shape:
            if isinstance(dim, tuple):
                dim = max(dim)
            if dim is not None and dim > max_dim:
                max_dim = dim
            if dim is not None and dim < min_dim:
                min_dim = dim
    if min_dim == float('inf'):
        min_dim = 0
    return max_dim, min_dim

count = 0
for model_file in correct_files:
    if count <= 1682 and model_file.endswith('.h5'):
        model_path = os.path.join(model_directory, model_file)
        try:
            model = tf.keras.models.load_model(model_path)
        
            layer_count = {f"Count{layer}": 0 for layer in keras_layers}
            activation_count = {f"Count{activation}": 0 for activation in keras_activations}
            regularization_count = {f"Count{regularization}": 0 for regularization in keras_regularizations}
            
            total_params = 0
            trainable_params = 0
            num_neurons = []
            input_shapes = []
            output_shapes = []
            dropout_rates = []
            activation_presence = []
            layer_params = []
            input_output_mismatch_count = 0
            max_input_output_mismatch_count = 0

            for depth, layer in enumerate(model.layers):
                layer_type = layer.__class__.__name__
                if f"Count{layer_type}" in layer_count:
                    layer_count[f"Count{layer_type}"] += 1
                
                if hasattr(layer, 'activation'):
                    activation = layer.activation.__name__
                    if f"Count{activation}" in activation_count:
                        activation_count[f"Count{activation}"] += 1
                    activation_presence.append(1)
                else:
                    activation_presence.append(0)

                params = layer.count_params()
                total_params += params
                layer_params.append(params)
                if layer.trainable:
                    trainable_params += params
                
                if hasattr(layer, 'units'):
                    num_neurons.append(layer.units)
                elif hasattr(layer, 'filters'):
                    num_neurons.append(layer.filters)
                
                if isinstance(layer, tf.keras.layers.Dropout):
                    dropout_rates.append(layer.rate)
                
                if hasattr(layer, 'input_shape') and layer.input_shape:
                    input_shapes.append(layer.input_shape[1:])
                
                if hasattr(layer, 'output_shape') and layer.output_shape:
                    output_shapes.append(layer.output_shape[1:]) 

            for i in range(len(model.layers) - 1):
                current_output_shape = model.layers[i].output_shape[1:]
                next_input_shape = model.layers[i + 1].input_shape[1:]
                if current_output_shape is not None and next_input_shape is not None:
                    if current_output_shape != next_input_shape:
                        input_output_mismatch_count += 1

            combined_counts = {**layer_count, **activation_count, **regularization_count}
            combined_counts['Model_File'] = model_file
            combined_counts['Total_Params'] = total_params
            combined_counts['Trainable_Params'] = trainable_params
            combined_counts['Num_Neurons'] = len(num_neurons)
            combined_counts['Max_Neurons'] = max(num_neurons) if num_neurons else 0
            combined_counts['Min_Neurons'] = min(num_neurons) if num_neurons else 0

            max_dim_input, min_dim_input = get_max_min_dimension(input_shapes)
            max_dim_output, min_dim_output = get_max_min_dimension(output_shapes)

            combined_counts['Max_Dimension_Input'] = max_dim_input
            combined_counts['Min_Dimension_Input'] = min_dim_input
            combined_counts['Max_Dimension_Output'] = max_dim_output
            combined_counts['Min_Dimension_Output'] = min_dim_output

            if max_dim_input != max_dim_output:
                max_input_output_mismatch_count += 1
        
            combined_counts['Max_Input_Output_Dimension_Mismatch_Count'] = max_input_output_mismatch_count
            combined_counts['Max_Dropout_Rate'] = max(dropout_rates) if dropout_rates else 0
            combined_counts['Min_Dropout_Rate'] = min(dropout_rates) if dropout_rates else 0
            combined_counts['Number_of_Layers'] = len(model.layers)
            combined_counts['Layer_Diversity'] = len([layer for layer in layer_count.values() if layer > 0])
            combined_counts['Activation_Presence'] = sum(activation_presence) / len(model.layers) if model.layers else 0
            combined_counts['Avg_Params_Per_Layer'] = total_params / len(model.layers) if model.layers else 0
            combined_counts['Input_Output_Mismatch_Count_Layer'] = input_output_mismatch_count
            
            model_info_df = pd.DataFrame([combined_counts])
            all_models_features_df = pd.concat([all_models_features_df, model_info_df], ignore_index=True)
            print(f"Processed {model_file}")
            count += 1
        except Exception as e:
            print(f"Error processing {model_file}: {e}")

output_csv_path = os.path.join(dest_directory, 'layer_activation_counts_correct.csv')
all_models_features_df.to_csv(output_csv_path, index=False)
print(f"Saved results to {output_csv_path}")

import os
import random
import pandas as pd
import tensorflow as tf
import numpy as np

model_directory = "D:\\ICSE_Dataset\\H5AllBuggy"
dest_directory = "."
if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)

buggy_files = os.listdir('D:\ICSE_Dataset\H5AllBuggy')
buggy_files = [file for file in buggy_files if 'Layer' in file or 'Activation' in file]
all_models_features_df = pd.DataFrame()

keras_layers = [
    "Dense", "Activation", "Dropout", "Flatten", "InputLayer",
    "Conv1D", "Conv2D", "Conv3D",
    "MaxPooling1D", "MaxPooling2D", "MaxPooling3D",
    "SimpleRNN", "GRU", "LSTM",
    "Embedding", "BatchNormalization", "LayerNormalization"
]

keras_activations = [
    "softmax", "relu", "tanh", "sigmoid", "hard_sigmoid",
    "exponential", "linear"
]

keras_regularizations = [
    "L1", "L2", "L1L2"
]

def get_max_min_dimension(shape_list):
    max_dim = 0
    min_dim = float('inf')
    for shape in shape_list:
        for dim in shape:
            if isinstance(dim, tuple):
                dim = max(dim)
            if dim is not None and dim > max_dim:
                max_dim = dim
            if dim is not None and dim < min_dim:
                min_dim = dim
    if min_dim == float('inf'):
        min_dim = 0
    return max_dim, min_dim

count = 0
for model_file in buggy_files:
    if count <= 1682 and model_file.endswith('.h5'):
        model_path = os.path.join(model_directory, model_file)
        try:
            model = tf.keras.models.load_model(model_path)
        
            layer_count = {f"Count{layer}": 0 for layer in keras_layers}
            activation_count = {f"Count{activation}": 0 for activation in keras_activations}
            regularization_count = {f"Count{regularization}": 0 for regularization in keras_regularizations}
            
            total_params = 0
            trainable_params = 0
            num_neurons = []
            input_shapes = []
            output_shapes = []
            dropout_rates = []
            activation_presence = []
            layer_params = []
            input_output_mismatch_count = 0
            max_input_output_mismatch_count = 0

            for depth, layer in enumerate(model.layers):
                layer_type = layer.__class__.__name__
                if f"Count{layer_type}" in layer_count:
                    layer_count[f"Count{layer_type}"] += 1
                
                if hasattr(layer, 'activation'):
                    activation = layer.activation.__name__
                    if f"Count{activation}" in activation_count:
                        activation_count[f"Count{activation}"] += 1
                    activation_presence.append(1)
                else:
                    activation_presence.append(0)

                params = layer.count_params()
                total_params += params
                layer_params.append(params)
                if layer.trainable:
                    trainable_params += params
                
                if hasattr(layer, 'units'):
                    num_neurons.append(layer.units)
                elif hasattr(layer, 'filters'):
                    num_neurons.append(layer.filters)
                
                if isinstance(layer, tf.keras.layers.Dropout):
                    dropout_rates.append(layer.rate)
                
                if hasattr(layer, 'input_shape') and layer.input_shape:
                    input_shapes.append(layer.input_shape[1:])
                
                if hasattr(layer, 'output_shape') and layer.output_shape:
                    output_shapes.append(layer.output_shape[1:]) 

            for i in range(len(model.layers) - 1):
                current_output_shape = model.layers[i].output_shape[1:]
                next_input_shape = model.layers[i + 1].input_shape[1:]
                if current_output_shape is not None and next_input_shape is not None:
                    if current_output_shape != next_input_shape:
                        input_output_mismatch_count += 1

            combined_counts = {**layer_count, **activation_count, **regularization_count}
            combined_counts['Model_File'] = model_file
            combined_counts['Total_Params'] = total_params
            combined_counts['Trainable_Params'] = trainable_params
            combined_counts['Num_Neurons'] = len(num_neurons)
            combined_counts['Max_Neurons'] = max(num_neurons) if num_neurons else 0
            combined_counts['Min_Neurons'] = min(num_neurons) if num_neurons else 0

            max_dim_input, min_dim_input = get_max_min_dimension(input_shapes)
            max_dim_output, min_dim_output = get_max_min_dimension(output_shapes)

            combined_counts['Max_Dimension_Input'] = max_dim_input
            combined_counts['Min_Dimension_Input'] = min_dim_input
            combined_counts['Max_Dimension_Output'] = max_dim_output
            combined_counts['Min_Dimension_Output'] = min_dim_output

            if max_dim_input != max_dim_output:
                max_input_output_mismatch_count += 1
        
            combined_counts['Max_Input_Output_Dimension_Mismatch_Count'] = max_input_output_mismatch_count
            combined_counts['Max_Dropout_Rate'] = max(dropout_rates) if dropout_rates else 0
            combined_counts['Min_Dropout_Rate'] = min(dropout_rates) if dropout_rates else 0
            combined_counts['Number_of_Layers'] = len(model.layers)
            combined_counts['Layer_Diversity'] = len([layer for layer in layer_count.values() if layer > 0])
            combined_counts['Activation_Presence'] = sum(activation_presence) / len(model.layers) if model.layers else 0
            combined_counts['Avg_Params_Per_Layer'] = total_params / len(model.layers) if model.layers else 0
            combined_counts['Input_Output_Mismatch_Count_Layer'] = input_output_mismatch_count
            
            model_info_df = pd.DataFrame([combined_counts])
            all_models_features_df = pd.concat([all_models_features_df, model_info_df], ignore_index=True)
            print(f"Processed {model_file}")
            count += 1
        except Exception as e:
            print(f"Error processing {model_file}: {e}")

output_csv_path = os.path.join(dest_directory, 'layer_activation_counts_buggy.csv')
all_models_features_df.to_csv(output_csv_path, index=False)
print(f"Saved results to {output_csv_path}")


layer_activation_buggy = pd.read_csv('layer_activation_counts_buggy.csv')
layer_activation_correct = pd.read_csv('layer_activation_counts_correct.csv')
layer_activation_buggy['Buggy'] = 1
layer_activation_correct['Buggy'] = 0
layer_activation_all = pd.concat([layer_activation_buggy, layer_activation_correct], ignore_index=True)
#layer_activation_all = layer_activation_all.drop(columns=['Model_File'])

layer_activation_df = layer_activation_all.copy()
layer_activation_df.to_csv('layer_activation_df.csv', index=False)