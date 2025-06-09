# Step 2: Model Architecture (updated with node aggregation)
import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import LabelEncoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

class PositionEmbedding(nn.Module):
    def __init__(self, dh, max_len=100):
        super().__init__()
        self.dh = dh
        self.max_len = max_len

        # Create positional encodings upfront
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dh, 2).float() * (-math.log(10000.0) / dh))
        pe = torch.zeros(max_len, dh)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # This makes it work with both CPU and GPU

    def forward(self, positions):
        # positions: [batch_size, num_nodes]
        # We'll use the node positions to index into our precomputed embeddings
        return self.pe[positions]  # [batch_size, num_nodes, dh]


class GraphTransformerLayer(nn.Module):
    def __init__(self, dh):
        super().__init__()
        self.dh = dh
        self.Wq = nn.Linear(dh, dh)
        self.Wk = nn.Linear(dh, dh)
        self.Wv = nn.Linear(dh, dh)
        self.ffn = nn.Sequential(
            nn.Linear(dh, dh * 2),
            nn.ReLU(),
            nn.Linear(dh * 2, dh)
        )
        self.norm1 = nn.LayerNorm(dh)
        self.norm2 = nn.LayerNorm(dh)

    def forward(self, H, adj_hop):
        # H: [batch_size, num_nodes, dh]
        # adj_hop: tuple of (adjacency, hop_emb)
        adj, hop_emb = adj_hop

        Q = self.Wq(H)
        K = self.Wk(H)
        V = self.Wv(H)

        # Compute attention scores with hop distance
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dh)

        # Incorporate adjacency and hop distance
        adj_expanded = adj.unsqueeze(-1).expand(-1, -1, -1, self.dh)
        hop_effect = (adj_expanded * hop_emb).mean(-1)  # [bs, n, n]
        attn_scores = attn_scores * hop_effect

        attn = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn, V)
        attn_out = self.norm1(attn_out + H)

        ffn_out = self.ffn(attn_out)
        out = self.norm2(ffn_out + attn_out)
        return out


class GraphBERT(nn.Module):
    def __init__(self, input_dim, dh=256, num_layers=3, num_classes=2, max_hop=20):
        super().__init__()
        self.dh = dh
        self.max_hop = max_hop

        # Feature embeddings
        self.feature_embed = nn.Sequential(
            nn.Linear(input_dim, dh),
            nn.ReLU(),
            nn.LayerNorm(dh)
        )

        # Structural embeddings
        self.wl_embed = nn.Embedding(5000, dh)
        self.intimacy_embed = nn.Sequential(
            nn.Linear(1, dh),
            nn.ReLU(),
            nn.LayerNorm(dh)
        )
        self.hop_embed = nn.Embedding(max_hop + 2, dh)  # +2 for -1 (disconnected) and 0 (self)
        self.pos_embed = PositionEmbedding(dh)

        # Transformer layers
        self.layers = nn.ModuleList([GraphTransformerLayer(dh) for _ in range(num_layers)])

        # Aggregation and prediction
        self.aggregate = nn.Sequential(
            nn.Linear(dh * 2, dh),
            nn.ReLU(),
            nn.LayerNorm(dh)
        )
        self.predict = nn.Sequential(
            nn.Linear(dh, dh // 2),
            nn.ReLU(),
            nn.Linear(dh // 2, num_classes)
        )

    def forward(self, X, wl, intimacy, hop, adj):
        # Input shapes:
        # X: [batch_size, num_nodes, input_dim]
        # wl: [batch_size, num_nodes]
        # intimacy: [batch_size, num_nodes]
        # hop: [batch_size, num_nodes, num_nodes]
        # adj: [batch_size, num_nodes, num_nodes]

        batch_size, num_nodes, _ = X.shape

        # Embed all features
        feature_emb = self.feature_embed(X)  # [bs, n, dh]
        wl_emb = self.wl_embed(wl)  # [bs, n, dh]
        intimacy_emb = self.intimacy_embed(intimacy.unsqueeze(-1))  # [bs, n, dh]

        # Prepare hop distance embedding
        hop_clamped = torch.clamp(hop, -1, self.max_hop) + 1  # Shift -1 to 0
        hop_emb = self.hop_embed(hop_clamped)  # [bs, n, n, dh]

        # Combine initial embeddings
        h = feature_emb + wl_emb + intimacy_emb  # [bs, n, dh]

        # Add positional encoding
        positions = torch.arange(num_nodes, device=h.device).unsqueeze(0).expand(batch_size, -1)  # [bs, n]
        h = h + self.pos_embed(positions)  # [bs, n, dh]
        # In GraphBERT.forward:
        for layer in self.layers:
            h = layer(h, (adj, hop_emb))  # Pass both adj and hop_emb

        # Aggregate node embeddings
        target_node = h[:, -1, :]  # Target node embedding
        neighbor_mean = h[:, :-1, :].mean(1)  # Mean of neighbor embeddings
        aggregated = torch.cat([target_node, neighbor_mean], dim=-1)  # [bs, dh*2]
        aggregated = self.aggregate(aggregated)  # [bs, dh]

        # Final prediction
        return self.predict(aggregated)  # [bs, num_classes]

def load_graph_data(data_dir="./output_graph_results"):
    print("Loading precomputed graph data...")
    return {
        'wl_codes': pd.read_csv(f"{data_dir}/wl_codes.csv", index_col=0)['wl_code'].to_dict(),
        'adj_matrix': np.load(f"{data_dir}/adj_matrix.npy"),
        'intimacy_matrix': np.load(f"{data_dir}/intimacy_matrix.npy"),
        'features': pd.read_csv(f"{data_dir}/protein_features.csv", index_col=0),
        'node_map': pd.read_csv(f"{data_dir}/node_mapping.csv"),
        'max_wl_code': np.load(f"{data_dir}/max_wl_code.npy").item(),
        'hop_matrix': np.load(f"{data_dir}/hop_matrix.npy")
    }
