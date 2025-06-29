import pandas as pd
import numpy as np
import torch
import networkx as nx
from sklearn.metrics import roc_curve
from data_loading_model import GraphBERT  # or GraphTransformer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # <-- Add this

import torch
device = 'cpu'  # Force CPU

from data_loading_model import GraphBERT

model = GraphBERT(input_dim=2048).to(device)

def compute_optimal_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]

def calculate_intimacy(feat1, feat2):
    # Cosine similarity as intimacy
    intimacy = np.dot(feat1, feat2) / (np.linalg.norm(feat1)*np.linalg.norm(feat2) + 1e-8)
    intimacy = np.clip(intimacy, 0.1, 0.9)
    return intimacy

def build_graph(edge_list_file):
    # Load edge list and create graph
    edges = pd.read_csv(edge_list_file)
    G = nx.from_pandas_edgelist(edges, 'protein1', 'protein2')
    return G

def calculate_hop_distance(G, protein1, protein2):
    try:
        return nx.shortest_path_length(G, source=protein1, target=protein2)
    except nx.NetworkXNoPath:
        return 5  # Assume maximum distance if no path

def assign_wl_code(G):
    wl_code_dict = {}
    wl_labels = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(G, node_attr=None)
    for idx, node in enumerate(G.nodes()):
        wl_code_dict[node] = idx % 1000  # Dummy: Map node to a number
    return wl_code_dict

def predict_DEG_pairs(model, G, deg_list, feature_dict, device='cuda'):
    model.eval()
    results = []
    wl_code_dict = assign_wl_code(G)
    
    pairs = []
    for i in range(len(deg_list)):
        for j in range(i+1, len(deg_list)):
            pairs.append((deg_list[i], deg_list[j]))

    with torch.no_grad():
        for p1, p2 in pairs:
            feat1 = feature_dict.get(p1, np.zeros(2048))
            feat2 = feature_dict.get(p2, np.zeros(2048))

            intimacy = calculate_intimacy(feat1, feat2)
            hop = calculate_hop_distance(G, p1, p2)

            X = torch.FloatTensor([feat1, feat2]).unsqueeze(0).to(device)
            wl = torch.LongTensor([[wl_code_dict.get(p1, 0), wl_code_dict.get(p2, 0)]]).to(device)
            intimacy_t = torch.FloatTensor([[intimacy, intimacy]]).to(device)
            hop_t = torch.LongTensor([[hop, hop]]).to(device)

            output = model(X, wl, intimacy_t, hop_t)
            prob = torch.softmax(output, dim=1)[0,1].item()

            results.append({
                'protein1': p1,
                'protein2': p2,
                'prob_interaction': prob
            })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = GraphBERT(input_dim=2048).to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))

    # Load original full graph
    G = build_graph("full_graph_edges.csv")  # Edge list

    # Load DEG list
    deg_df = pd.read_csv("deg_list.csv")
    deg_list = deg_df['protein_id'].tolist()

    # Load features
    features_df = pd.read_csv("/mnt/data/shyam/kavya/project/pair_embeddings_with_proteinids.csv", index_col=0)
    feature_dict = {idx: row.values for idx, row in features_df.iterrows()}

    # Predict
    predictions = predict_DEG_pairs(model, G, deg_list, feature_dict, device)

    # Save
    predictions.to_csv("deg_pair_predictions.csv", index=False)
    print("Predictions for DEG pairs saved!")

