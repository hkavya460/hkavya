This directory contains the core source code for graph-based protein interaction modeling using transformer (BERT) techniques. 
The pipeline includes data preprocessing, graph construction, model training, and testing. Below is a brief description of each file:

**1. data_loading.py**
Purpose: Handles loading and preprocessing of protein sequence data.
Details:Reads raw sequence datasets.
Performs any necessary cleaning, tokenization, or formatting.
Prepares data for graph construction and modeling steps.

**2. Graph_construction.py**
Purpose: Constructs a protein interaction graph where nodes represent protein pairs and edges are formed when nodes share a protein.
Details:
Converts protein sequences into vector representations.
Forms pairwise embeddings of proteins.
Builds a graph structure based on shared proteins.
Calculates positional encodings, adjacency matrix, and other features required by the BERT model.
Outputs: Graph-related data files used in subsequent model training, stored in the output_results/ directory.

**3. model_training.py**
Purpose: Trains the transformer-based model on the constructed graph data.
Details:
Initializes the model using BERT-based architecture.
Feeds in node features, adjacency matrices, and positional encodings.
Handles training loop, loss calculation, and optimization.

**4. model_testing.py**
Purpose: Evaluates the trained model's performance on test data.
Loads the trained model and test graph data.
Computes predictions and evaluation metrics (accuracy, precision, recall).
Outputs model performance statistics.
