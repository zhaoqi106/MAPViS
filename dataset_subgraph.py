import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy

import pandas as pd
import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
import pickle

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def read_smiles(data_path):
    df = pd.read_csv(data_path, sep=',')
    smiles_data = df['smiles']
    labels_1 = df['k_100']
    labels_2 = df['k_1000']
    labels_3 = df['k_10000']
    return smiles_data, labels_1, labels_2, labels_3
    # mol = Chem.MolFromSmiles(smiles)
    # if mol != None:
    #     smiles_data.append(smiles)
    # return smiles_data


"""
Remove a connected subgraph from the original molecule graph. 
Args:
    1. Original graph (networkx graph)
    2. Index of the starting atom from which the removal begins (int)
    3. Percentage of the number of atoms to be removed from original graph

Outputs:
    1. Resulting graph after subgraph removal (networkx graph)
    2. Indices of the removed atoms (list)
"""


def removeSubgraph(Graph, center, percent=0.2):
    assert percent <= 1
    G = Graph.copy()
    num = int(np.floor(len(G.nodes) * percent))
    removed = []
    temp = [center]

    while len(removed) < num and temp:
        neighbors = []
        for n in temp:
            neighbors.extend([i for i in G.neighbors(n) if i not in temp and i not in removed])
        for n in temp:
            if len(removed) < num:
                G.remove_node(n)
                removed.append(n)
            else:
                break
        temp = list(set(neighbors))
    return G, removed

from torch_geometric.data import Dataset as GeometricDataset, Data
class MoleculeDataset(Dataset):
    def __init__(self, data_path):
        super(Dataset, self).__init__()
        # self.smiles_data = read_smiles(data_path)
        self.smiles_data, self.label1, self.label2, self.label3 = read_smiles(data_path)
        self.data_path = data_path


    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        # mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        # Sample 2 different centers to start for i and j
        start_xi, start_j = random.sample(list(range(N)), 2)

        # Construct the original molecular graph from edges (bonds)
        edges = []
        for bond in bonds:
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        molGraph = nx.Graph(edges)

        # Get the graph for i and j after removing subgraphs
        # G_i, removed_i = removeSubgraph(molGraph, start_i)
        # G_j, removed_j = removeSubgraph(molGraph, start_j)

        # percent_i, percent_j = random.uniform(0, 0.25), random.uniform(0, 0.25)
        percent_xi, percent_j = 0.25, 0.25
        # percent_i, percent_j = 0.2, 0.2
        G_xi, removed_xi = removeSubgraph(molGraph, start_xi, percent_xi)
        # G_j, removed_j = removeSubgraph(molGraph, start_j, percent_j)

        for atom in atoms:
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)
        # x shape (N, 2) [type, chirality]

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            # edge_type += 2 * [MOL_BONDS[bond.GetBondType()]]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        if len(edge_feat) == 0:
            print(f"edge_attr{index} is y empty")
            edge_attr = torch.empty((0, 2), dtype=torch.long)
        else:
            edge_attr = torch.tensor(edge_feat, dtype=torch.long)

        # x shape (N, 2) [type, chirality]

        # Mask the atoms in the removed list
        x_xi = deepcopy(x)
        for atom_idx in removed_xi:
            # Change atom type to 118, and chirality to 0
            x_xi[atom_idx, :] = torch.tensor([len(ATOM_LIST), 0])
        # x_j = deepcopy(x)
        # for atom_idx in removed_j:
        #     # Change atom type to 118, and chirality to 0
        #     x_j[atom_idx, :] = torch.tensor([len(ATOM_LIST), 0])

        # Only consider bond still exist after removing subgraph
        row_xi, col_xi, row_j, col_j = [], [], [], []
        edge_feat_xi, edge_feat_j = [], []
        G_xi_edges = list(G_xi.edges)
        # G_j_edges = list(G_j.edges)
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feature = [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ]
            if (start, end) in G_xi_edges:
                row_xi += [start, end]
                col_xi += [end, start]
                edge_feat_xi.append(feature)
                edge_feat_xi.append(feature)
            # if (start, end) in G_j_edges:
            #     row_j += [start, end]
            #     col_j += [end, start]
            #     edge_feat_j.append(feature)
            #     edge_feat_j.append(feature)

        edge_index_xi = torch.tensor([row_xi, col_xi], dtype=torch.long)
        if len(edge_feat_xi) == 0:
            print(f"edge_feat_xi{index} is m empty")
            edge_attr_xi = torch.empty((0, 2), dtype=torch.long)
        else:
            edge_attr_xi = torch.tensor(edge_feat_xi, dtype=torch.long)
        # edge_index_j = torch.tensor([row_j, col_j], dtype=torch.long)
        # edge_attr_j = torch.tensor(np.array(edge_feat_j), dtype=torch.long)

        y1 = torch.tensor(self.label1[index], dtype=torch.int64)
        y2 = torch.tensor(self.label2[index], dtype=torch.int64)
        y3 = torch.tensor(self.label3[index], dtype=torch.int64)

        data_xi = Data(x=x_xi, edge_index=edge_index_xi, edge_attr=edge_attr_xi)
        # data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y1=y1, y2=y2, y3=y3)

        """"""""""""""""""""""""""""""""
        """
        # Get the graph for i and j after removing subgraphs
        start_i, start_j = random.sample(list(range(N)), 2)
        percent_i, percent_j = 0.2, 0.2
        G_i, removed_i = removeSubgraph(molGraph, start_i, percent=percent_i)
        G_j, removed_j = removeSubgraph(molGraph, start_j, percent=percent_j)

        atom_remain_indices_i = [i for i in range(N) if i not in removed_i]
        atom_remain_indices_j = [i for i in range(N) if i not in removed_j]

        # Only consider bond still exist after removing subgraph
        row_i, col_i, row_j, col_j = [], [], [], []
        edge_feat_i, edge_feat_j = [], []
        G_i_edges = list(G_i.edges)
        G_j_edges = list(G_j.edges)

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feature = [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ]
            if (start, end) in G_i_edges or (end, start) in G_i_edges:
                row_i += [start, end]
                col_i += [end, start]
                edge_feat_i.append(feature)
                edge_feat_i.append(feature)
            if (start, end) in G_j_edges or (end, start) in G_j_edges:
                row_j += [start, end]
                col_j += [end, start]
                edge_feat_j.append(feature)
                edge_feat_j.append(feature)

        edge_index_i = torch.tensor([row_i, col_i], dtype=torch.long)
        edge_attr_i = torch.tensor(np.array(edge_feat_i), dtype=torch.long)
        edge_index_j = torch.tensor([row_j, col_j], dtype=torch.long)
        edge_attr_j = torch.tensor(np.array(edge_feat_j), dtype=torch.long)

        ############################
        # Random Atom/Edge Masking #
        ############################

        num_mask_nodes_i = max([0, math.floor(0.15 * N) - len(removed_i)])
        num_mask_edges_i = max([0, edge_attr_i.size(0) // 2 - math.ceil(0.85 * M)])
        num_mask_nodes_j = max([0, math.floor(0.15 * N) - len(removed_j)])
        num_mask_edges_j = max([0, edge_attr_j.size(0) // 2 - math.ceil(0.85 * M)])
        mask_nodes_i = random.sample(atom_remain_indices_i, num_mask_nodes_i)
        mask_nodes_j = random.sample(atom_remain_indices_j, num_mask_nodes_j)
        mask_edges_i_single = random.sample(list(range(edge_attr_i.size(0) // 2)), num_mask_edges_i)
        mask_edges_j_single = random.sample(list(range(edge_attr_j.size(0) // 2)), num_mask_edges_j)

        mask_edges_i = [2 * i for i in mask_edges_i_single] + [2 * i + 1 for i in mask_edges_i_single]
        mask_edges_j = [2 * i for i in mask_edges_j_single] + [2 * i + 1 for i in mask_edges_j_single]

        x_i = deepcopy(x)
        for atom_idx in range(N):
            if (atom_idx in mask_nodes_i) or (atom_idx in removed_i):
                x_i[atom_idx, :] = torch.tensor([len(ATOM_LIST), 0])
        edge_index_final_i = torch.zeros((2, edge_attr_i.size(0) - 2 * num_mask_edges_i), dtype=torch.long)
        edge_attr_final_i = torch.zeros((edge_attr_i.size(0) - 2 * num_mask_edges_i, 2), dtype=torch.long)
        count = 0
        for bond_idx in range(edge_attr_i.size(0)):
            if bond_idx not in mask_edges_i:
                edge_index_final_i[:, count] = edge_index_i[:, bond_idx]
                edge_attr_final_i[count, :] = edge_attr_i[bond_idx, :]
                count += 1
        data_i = Data(x=x_i, edge_index=edge_index_final_i, edge_attr=edge_attr_final_i)

        x_j = deepcopy(x)
        for atom_idx in range(N):
            if (atom_idx in mask_nodes_j) or (atom_idx in removed_j):
                x_j[atom_idx, :] = torch.tensor([len(ATOM_LIST), 0])
        edge_index_final_j = torch.zeros((2, edge_attr_j.size(0) - 2 * num_mask_edges_j), dtype=torch.long)
        edge_attr_final_j = torch.zeros((edge_attr_j.size(0) - 2 * num_mask_edges_j, 2), dtype=torch.long)
        count = 0
        for bond_idx in range(edge_attr_j.size(0)):
            if bond_idx not in mask_edges_j:
                edge_index_final_j[:, count] = edge_index_j[:, bond_idx]
                edge_attr_final_j[count, :] = edge_attr_j[bond_idx, :]
                count += 1
        data_j = Data(x=x_j, edge_index=edge_index_final_j, edge_attr=edge_attr_final_j)
        """

        return data, data_xi


    def __len__(self):
        return len(self.smiles_data)
        # return len(self.processed_data)


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size

    def get_data_loaders(self, batch_size=None):
        batch_size = batch_size if batch_size else self.batch_size
        dataset = MoleculeDataset(data_path=self.data_path)
        train_loader, valid_loader = self.get_train_validation_data_loaders(dataset,batch_size)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset,batch_size):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))

        # random_state = np.random.RandomState(seed=666)
        # random_state.shuffle(indices)
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, pin_memory=True, persistent_workers=True)

        valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True, pin_memory=True, persistent_workers=True)

        return train_loader, valid_loader


if __name__ == "__main__":
    # data_path = 'data/chem_dataset/zinc_standard_agent/processed/smiles.csv'
    # data_path ='../data/pretrain_data/data10m.csv'
    data_path = '../data/downstream_data/1.csv'
    # data_path = os.path.abspath('../data/pretrain_data/data10m.csv')
    # print("Absolute data path:", data_path)

    # dataset = MoleculeDataset(data_path=data_path)
    # print(dataset)
    # print(dataset.__getitem__(0))

    dataset = MoleculeDatasetWrapper(batch_size=4, num_workers=4, valid_size=0.1, data_path=data_path)

    print("okok  dataset to train_loader, valid_loader  ")
    train_loader, valid_loader = dataset.get_data_loaders()
    print(" train_loader, valid_loader  ok")
    for bn, (data, data_xi, data_i, data_j) in enumerate(train_loader):
        print(data, data_xi, data_i, data_j)
        break
