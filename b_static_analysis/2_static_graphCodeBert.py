import tensorflow as tf
import json
import os
import pandas as pd
from transformers import RobertaModel, RobertaTokenizer
import torch
import warnings

warnings.filterwarnings("ignore")

model_directory = "D:\\ICSE_Dataset\\H5AllBuggy"
output_directory = "D:\\ICSE_Dataset\\Derived_Features\\embedding_features_buggy"
log_file = "D:\\ICSE_Dataset\\Derived_Features\\error_bert_log1.txt"
os.makedirs(output_directory, exist_ok=True)

graphcodebert_model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")

def model_to_text(model_config):
    text_representation = json.dumps(model_config)
    return text_representation

def extract_graphcodebert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = graphcodebert_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embedding

with open(log_file, 'w') as log:
    for model_file in os.listdir(model_directory):
        if model_file.endswith('.h5'):
            model_path = os.path.join(model_directory, model_file)
            try:
                model = tf.keras.models.load_model(model_path)
                model_config = model.get_config()
                text_representation = model_to_text(model_config)
                embedding = extract_graphcodebert_embedding(text_representation)
                output_data = {
                    'model_file': model_file,
                    'embedding': embedding.flatten().tolist()
                }
                output_json_path = os.path.join(output_directory, f'{model_file}.json')
                with open(output_json_path, 'w') as json_file:
                    json.dump(output_data, json_file, indent=4)
                
                print(f'Saved embeddings for {model_file} to {output_json_path}')
            except Exception as e:
                log.write(f'Error processing {model_file}: {e}\n')
                print(f'Error processing {model_file}, logged error and moving to next file.')

print('Processing done!! Check error_log.txt for any errors.')