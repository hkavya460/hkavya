# training.py
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from data_loading_model_building import GraphBERT, load_graph_data
import torch.nn as nn 
import networkx as nx 
import pandas as pd 

class PPIDataset:
    def __init__(self, graph_data):
        self.graph_data = graph_data
        self.node_to_idx = {row['node']: row['index'] for _, row in graph_data['node_map'].iterrows()}
        self.idx_to_node = {v: k for k, v in self.node_to_idx.items()}
        self.nodes = list(self.node_to_idx.keys())
        
    def sample_subgraph(self, target_node, k=10):
        target_idx = self.node_to_idx[target_node]
        intimacy_scores = self.graph_data['intimacy_matrix'][target_idx]
        top_k = np.argsort(intimacy_scores)[-k-1:-1]  # Exclude self
        subgraph_nodes = [self.idx_to_node[i] for i in top_k] + [target_node]
        
        # Get features
        X = [self.graph_data['features'].loc[node].values for node in subgraph_nodes]
        wl = [self.graph_data['wl_codes'][node] for node in subgraph_nodes]
        intimacy = [intimacy_scores[self.node_to_idx[node]] for node in subgraph_nodes]
        
        # Compute hop distances
        adj = self.graph_data['adj_matrix']
        hops = []
        for node in subgraph_nodes:
            node_idx = self.node_to_idx[node]
            hop_dist = nx.shortest_path_length(
                nx.from_numpy_array(adj), 
                source=target_idx, 
                target=node_idx
            ) if node_idx != target_idx else 0
            hops.append(hop_dist)
            
        return (
            torch.FloatTensor(np.array(X)),
            torch.LongTensor(np.array(wl)),
            torch.FloatTensor(np.array(intimacy)),
            torch.FloatTensor(np.array(hops))
        )

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_data = load_graph_data()
    dataset = PPIDataset(graph_data)
    label_df =  pd.read_csv(f"{graph_data[data_dir]}/node_label.csv",index_col=0)
    # Split data
    nodes = dataset.nodes
    labels = label_df['label'].values   # Replace with your actual labels

    train_nodes, test_nodes, train_labels, test_labels = train_test_split(
        nodes, labels, test_size=0.2, random_state=42
    )
    
    # Initialize model
    input_dim = len(graph_data['features'].iloc[0])
    model = GraphBERT(input_dim,max_wl_code =graph_data['max_wl_code').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(100):
        model.train()
        total_loss = correct = 0
        
        for node, label in zip(train_nodes, train_labels):
            optimizer.zero_grad()
            X, wl, intimacy, hop = dataset.sample_subgraph(node)
            X, wl, intimacy, hop = X.to(device), wl.to(device), intimacy.to(device), hop.to(device)
            
            output = model(X, wl, intimacy, hop)
            loss = criterion(output, torch.LongTensor([label]).to(device))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (output.argmax() == label).sum().item()
        
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_nodes):.4f}, Acc={correct/len(train_nodes):.4f}")
    
    # Evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for node, label in zip(test_nodes, test_labels):
            X, wl, intimacy, hop = dataset.sample_subgraph(node)
            X, wl, intimacy, hop = X.to(device), wl.to(device), intimacy.to(device), hop.to(device)
            output = model(X, wl, intimacy, hop)
            correct += (output.argmax() == label).sum().item()
    
    print(f"Test Accuracy: {correct/len(test_nodes):.4f}")

if __name__ == "__main__":
    train()
