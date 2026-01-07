import csv
import pandas as pd
import numpy as np
from rdkit import Chem
import networkx as nx


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms() 
    features = []  
    for atom in mol.GetAtoms():
        feature = atom_features(atom)  
        features.append(feature / sum(feature))
    edges = []  
    for bond in mol.GetBonds():  
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()  
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index  



def process_data(drug_synergy_dataset, drug_smiles, cell_gene_expressions):
    cell_features = []
    with open(cell_gene_expressions) as csvfile:
        csv_reader = csv.reader(csvfile)  
        for row in csv_reader:
            cell_features.append(row)
    cell_features = np.array(cell_features)
    compound_iso_smiles = []
    df = pd.read_csv(drug_smiles)
    compound_iso_smiles += list(df['smile'])
    compound_iso_smiles = set(compound_iso_smiles)  
    smile_graph = {}

    for smile in compound_iso_smiles:
        if smile not in smile_graph: 
            smile_graph[smile] = {}
        g = smile_to_graph(smile)  
        smile_graph[smile] = g 

    df = pd.read_csv(drug_synergy_dataset)
    drug1, drug2, cell, label = list(df['drug1']), list(df['drug2']), list(df['cell']), list(df['label'])
    drug1, drug2, cell, label = np.asarray(drug1), np.asarray(drug2), np.asarray(cell), np.asarray(label)
    return drug1, drug2, cell, label, smile_graph, cell_features

