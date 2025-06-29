Deep learning approches for studying the protein interaction network in Rheumatod athritis (RA) 

This project focuses on understanding the molecular mechanisms of Rheumatoid Arthritis (RA), a chronic autoimmune disorder marked by immune dysregulation and inflammation. The heterogeneity of transcriptomic data across patients makes it challenging to identify consistent biomarkers or pathways.
To address this, we use a graph-based deep learning approach which is  transformer models to analyze transcriptomic data, cluster genes, perform functional enrichment, and predict **protein-protein interactions (PPIs)** using a **Graph-BERT** model.

##  Project Objectives
Analyze transcriptomic data (microarray and RNA-Seq) to identify differentially expressed genes (DEGs).
Cluster DEGs to detect co-regulated groups and perform functional enrichment analysis.
Construct a  biological interaction graph using gene and protein information.
Train a Graph-BERT  model on experimentally validated PPIs to learn contextual protein-pair embeddings.
Predict novel PPIs among RA-associated genes and visualize the resulting interaction network.

##  Methodology 
1. Transcriptomic Analysis
Raw expression data (microarray and RNA-Seq) is processed to identify differentially expressed genes (DEGs) between RA and control samples.
DEGs are clustered using:K-Means Clustering and Hierarchical Clustering**
Functional enrichment analysis of each cluster is performed using:
**STRING**(https://string-db.org)
**DAVID**(https://david.ncifcrf.gov)

### 2 Graph Construction 
Proteins/Genes are represented as pairs, each node  in the graph is a pair of proteins.An edge is created between nodes if the two nodes share a protein.
Node features include:Protein sequence embeddings (via **SeqVec**)
Positional encodings,Graph-theoretic features like adjacency matrix, hop distance,degree matrix, and Weisfeiler-Lehman (WL) codes.

### 3 PPI predication using the Graph-BERT
The **Graph-BERT model is used to encode the graph using transformer-based attention mechanisms tailored for graph structure.
Training data:
Positive PPI pairs from BioGRID 
Negative (non-interacting) pairs from Negatome
Protein sequences** are retrieved from UniProt and embedded using SeqVec
Trained model is evaluated on RA-specific DEGs to predict novel PPIs.
A final RA-specific interaction network is constructed from the predictions.



