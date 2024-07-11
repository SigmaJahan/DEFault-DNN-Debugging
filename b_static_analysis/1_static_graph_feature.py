import os
import pandas as pd
import numpy as np
import networkx as nx
import tensorflow as tf
import json

def build_graph(model):
    G = nx.DiGraph()
    for i, layer in enumerate(model.layers):
        G.add_node(i, layer_type=type(layer).__name__)
        if i > 0:
            G.add_edge(i - 1, i)
    return G

def compute_graph_features(G):
    graph_features = {}
    errors = {}

    try:
        graph_features['degree_centrality'] = nx.degree_centrality(G)
    except Exception as e:
        errors['degree_centrality'] = str(e)

    try:
        graph_features['closeness_centrality'] = nx.closeness_centrality(G)
    except Exception as e:
        errors['closeness_centrality'] = str(e)

    try:
        graph_features['betweenness_centrality'] = nx.betweenness_centrality(G)
    except Exception as e:
        errors['betweenness_centrality'] = str(e)

    try:
        graph_features['degree'] = dict(G.degree())
    except Exception as e:
        errors['degree'] = str(e)

    try:
        graph_features['transitivity'] = nx.transitivity(G)
    except Exception as e:
        errors['transitivity'] = str(e)

    try:
        graph_features['shortest_path_length'] = dict(nx.shortest_path_length(G))
    except Exception as e:
        errors['shortest_path_length'] = str(e)

    try:
        graph_features['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=1000)
    except Exception as e:
        errors['eigenvector_centrality'] = str(e)

    try:
        graph_features['pagerank'] = nx.pagerank(G)
    except Exception as e:
        errors['pagerank'] = str(e)

    try:
        hubs, authorities = nx.hits(G)
        graph_features['hubs'] = hubs
        graph_features['authorities'] = authorities
    except Exception as e:
        errors['hits'] = str(e)

    try:
        graph_features['k_core'] = dict(nx.core_number(G))
    except Exception as e:
        errors['k_core'] = str(e)

    try:
        graph_features['density'] = nx.density(G)
    except Exception as e:
        errors['density'] = str(e)

    return graph_features, errors

model_directory = "D:\\ICSE_Dataset\\H5AllBuggy"
dest_directory = "D:\\ICSE_Dataset\\Derived_Features\\graph_features"

if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)

for model_file in os.listdir(model_directory):
    if model_file.endswith('.h5'):
        model_path = os.path.join(model_directory, model_file)
        output_json_path = os.path.join(dest_directory, model_file.replace('.h5', '_graph_features.json'))
        error_log_path = os.path.join(dest_directory, model_file.replace('.h5', '_error_log.txt'))
    
        if os.path.exists(output_json_path):
            print(f"Skipping {model_file}, output file already exists.")
            continue
        
        try:
            model = tf.keras.models.load_model(model_path)
            G = build_graph(model)
            graph_features, errors = compute_graph_features(G)
            
            with open(output_json_path, 'w') as json_file:
                json.dump(graph_features, json_file, indent=4)
            
            if errors:
                with open(error_log_path, 'w') as log_file:
                    for error_key, error_message in errors.items():
                        log_file.write(f"{error_key}: {error_message}\n")
            
            print(f"Processed {model_file}")

        except Exception as e:
            print(f"Error processing {model_file}: {e}")

