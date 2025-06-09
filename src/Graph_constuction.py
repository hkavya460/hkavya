# graph_construction.py
import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import LabelEncoder
import math


class PPIGraphBuilder:
    def __init__(self, protein_pairs, protein_features, labels, df, output_dir="output_graph_results"):
        self.protein_pairs = protein_pairs
        self.protein_features = protein_features
        self.labels = labels
        self.df = df  # Store the dataframe
        self.nodes = list(set([p for pair in protein_pairs for p in pair]))
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(self.nodes)}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Build graph
        self.G = nx.Graph()
        self.G.add_nodes_from(self.nodes)
        self.G.add_edges_from(protein_pairs)

        # Compute and save embeddings
        self._compute_and_save_embeddings()

    def _compute_and_save_embeddings(self):
        print("Computing WL codes...")
        self.wl_codes = self._compute_wl_codes()
        pd.DataFrame.from_dict(self.wl_codes, orient='index').to_csv(
            f"{self.output_dir}/wl_codes.csv", header=['wl_code'])

        print("Computing adjacency matrix...")
        self.adj_matrix = self._build_adjacency_matrix()
        np.save(f"{self.output_dir}/adj_matrix.npy", self.adj_matrix)

        print("Computing degree matrix...")
        self.degree_matrix = self._build_degree_matrix()
        np.save(f"{self.output_dir}/degree_matrix.npy", self.degree_matrix)

        print("Computing intimacy matrix...")
        self.intimacy_matrix = self._compute_intimacy_matrix()
        np.save(f"{self.output_dir}/intimacy_matrix.npy", self.intimacy_matrix)

        print("Saving node features...")
        pd.DataFrame.from_dict(self.protein_features, orient='index').to_csv(
            f"{self.output_dir}/protein_features.csv")

        print("Saving node mappings...")
        pd.DataFrame({'node': self.nodes, 'index': range(len(self.nodes))}).to_csv(
            f"{self.output_dir}/node_mapping.csv", index=False)

        print("Saving label file...")
        label_map = self._save_labels()
        pd.DataFrame.from_dict(label_map, orient='index', columns=['label']).to_csv(
            f"{self.output_dir}/node_labels.csv")

    def _build_adjacency_matrix(self):
        n = len(self.nodes)
        adj = np.zeros((n, n))
        for i, j in self.protein_pairs:
            idx_i = self.node_to_idx[i]
            idx_j = self.node_to_idx[j]
            adj[idx_i][idx_j] = 1
            adj[idx_j][idx_i] = 1
        return adj

    def _build_degree_matrix(self):
        return np.diag(np.sum(self.adj_matrix, axis=1))

    def _compute_wl_codes(self):
        labels = {node: hash(str(self.protein_features[node])) for node in self.nodes}
        for _ in range(3):
            new_labels = {}
            for node in self.nodes:
                neighbors = sorted([labels[n] for n in self.G.neighbors(node)])
                new_labels[node] = hash(f"{labels[node]}_{'_'.join(map(str, neighbors))}")
            labels = new_labels
        le = LabelEncoder()
        encoded = le.fit_transform(list(labels.values()))

        # Save max WL code
        max_wl_code = max(encoded)
        np.save(f"{self.output_dir}/max_wl_code.npy", max_wl_code)

        return {node: encoded[i] for i, node in enumerate(self.nodes)}
      def _compute_intimacy_matrix(self, alpha=0.15):
        col_norm_adj = self.adj_matrix @ np.linalg.inv(self.degree_matrix)
        identity = np.eye(len(self.nodes))
        return alpha * np.linalg.inv(identity - (1 - alpha) * col_norm_adj)

    def _save_labels(self):
        """Create and return a mapping from protein to label"""
        label_map = {}
        for _, row in self.df.iterrows():  # Fixed typo here (was itterows)
            label_map[row['protein1']] = row['label']
            label_map[row['protein2']] = row['label']
        return label_map


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    protein_pairs = list(zip(df['protein1'], df['protein2']))
    labels = df['label'].values

    # Extract features (columns starting with 'emb_')
    feature_cols = [col for col in df.columns if col.startswith('emb_')]
    protein_features = {}
    for _, row in df.iterrows():
        protein_features[row['protein1']] = row[feature_cols].values
        protein_features[row['protein2']] = row[feature_cols].values

    return protein_pairs, protein_features, labels, df


if __name__ == "__main__":
    protein_pairs, protein_features, labels, df = load_data("balanced_data.csv")
    builder = PPIGraphBuilder(protein_pairs, protein_features, labels, df)
    print("Graph construction and embedding calculation completed!")



