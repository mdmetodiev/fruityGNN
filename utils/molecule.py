from dataclasses import dataclass, field
from typing import List, Tuple, Set
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDepictor
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch

bond_type_mapping = {Chem.rdchem.BondType.SINGLE: 1,
                     Chem.rdchem.BondType.DOUBLE: 2,
                     Chem.rdchem.BondType.TRIPLE: 3,
                     Chem.rdchem.BondType.AROMATIC: 4}


@dataclass
class Molecule:
    smiles:str
    _mol: Chem.rdchem.Mol = field(init=False)
    nodes: List[int] = field(init=False)
    edges:List[int] = field(init=False)
    adjacency_list:List[Tuple[int,int]] = field(init=False)

    def __post_init__(self):

        _mol = Chem.MolFromSmiles(self.smiles)
        _mol = Chem.AddHs(_mol)
        self._mol = _mol
        self.nodes = [atom.GetAtomicNum() for atom in self._mol.GetAtoms()]
        e, adj = self._create_edges_and_adjacency_list()
        self.edges = e
        self.adjacency_list = adj
  
    
    def _create_edges_and_adjacency_list(self):
        edges = []
        adjacency_list = []
        for bond in self._mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond_type_mapping.get(bond.GetBondType(), 0) 
            edges.append(bond_type)
            adjacency_list.append((j, i))  

        return edges, adjacency_list

    def creat_nx_graph(self) -> nx.Graph:
        G = nx.Graph()

        for i, atomic_num in enumerate(self.nodes):
            G.add_node(i, label=f"{atomic_num}")

        for idx, (i, j) in enumerate(self.adjacency_list):
            G.add_edge(i, j, bond=self.edges[idx])  

        return G
    
    def get_atomic_positions(self) :
        rdDepictor.Compute2DCoords(self._mol)
        atom_positions = {i: (self._mol.GetConformer().GetAtomPosition(i).x, self._mol.GetConformer().GetAtomPosition(i).y) for i in range(len(self.nodes))}
        return atom_positions

    def plot_molecule(self) -> None:

        plt.figure(figsize=(19, 6))
        
        G = self.creat_nx_graph()

        # Draw nodes with labels
        labels = {i: f"{G.nodes[i]['label']}" for i in G.nodes}
        positions = self.get_atomic_positions()

        nx.draw(G, positions, labels = labels, with_labels=True, node_color="lightblue", edge_color="black", node_size=1500, font_size=15)

        # Draw bond types as edge labels
        edge_labels = {(i, j): f"{self.edges[idx]}" for idx, (i, j) in enumerate(self.adjacency_list)}
        nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_labels, font_size=12, label_pos=0.5)

        plt.show()
        plt.close()



class MoleculeData:
    molecules: List[Molecule]
        
    def read_data(self, path:str) -> dict:

        data = pd.read_csv(path, delimiter=",")

        molecules = {"smiles":data["IsomericSMILES"].to_numpy(),
                     "fruity":data["fruity"].to_numpy()}
        return molecules



def read_data(path:str) -> dict:

    data = pd.read_csv(path, delimiter=",")

    molecules = {"smiles":data["IsomericSMILES"].to_numpy(),
                 "fruity":data["fruity"].to_numpy()}
    return molecules







