#BioGRID protein interaction pair downloaded from the website then processed the file to get protein id's
#Followed by sequence to vector conversion 

import matplotlib.pyplot as plt
import pandas as pd
from Bio import SeqIO

#Biogrid human protein interaction pair protein

positive_pppi_pair = pd.read_csv("~/PycharmProjects/python_project/DL/RA_PROJECT/BIOGRID-ORGANISM-4.4.242.tab3/BIOGRID-ORGANISM-Homo_sapiens-4.4.242.tab3.txt",sep="\t",index_col=False)
print(positive_pppi_pair.shape)
print(positive_pppi_pair.columns)

protein_ids_data =  positive_pppi_pair[['SWISS-PROT Accessions Interactor A','SWISS-PROT Accessions Interactor B', 'Official Symbol Interactor A','Official Symbol Interactor B','Throughput']]
print(protein_ids_data.shape)

protein_pair_data =  protein_ids_data[['SWISS-PROT Accessions Interactor A','SWISS-PROT Accessions Interactor B']]
print(protein_pair_data.nunique())

interactor_A =  protein_pair_data['SWISS-PROT Accessions Interactor A']
interactor_B =  protein_pair_data['SWISS-PROT Accessions Interactor B']
total_unique_proteins =  set(interactor_A.unique()).union(set(interactor_B.unique()))
print(len(total_unique_proteins))   #25335

total_unique_proteins_df = pd.DataFrame(total_unique_proteins)
print(total_unique_proteins_df.shape)
protein_pair_data.to_csv("./Biogrid_protein_pair.csv",sep="\t",index=False)

def extract_sequences(accession_ids, fasta_file):
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        uniprot_id = record.id.split("|")[1]  # Extracts Swiss-Prot Accession ID
        if uniprot_id in accession_ids:
            sequences[uniprot_id] = str(record.seq)
    return sequences

# Load your protein pairs file (assuming tab-separated)
protein_pairs_file = "/home/ibab/PycharmProjects/python_project/DL/RA_PROJECT/Biogrid_protein_pair.csv"
with open(protein_pairs_file) as f:
    accession_list = set(line.strip().split(",") for line in f)

fasta_file = "/home/ibab/PycharmProjects/python_project/DL/RA_PROJECT/uniprot_sprot.fasta"
seq_dict = extract_sequences(accession_list, fasta_file)

# Save extracted sequences
with open("extracted_sequences.fasta", "w") as out_f:
    for acc, seq in seq_dict.items():
        out_f.write(f">{acc}\n{seq}\n")

print("Sequences saved to extracted_sequences.fasta")

#######################################################################################################################################

#Negative interaction data processing and api calling 



