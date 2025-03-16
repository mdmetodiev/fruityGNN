from networkx import adjacency_data
import pandas as pd
from molecule import Molecule
from torch.utils.data import Dataset
from typing import List
import torch


class MoleculeData(Dataset):

    def __init__(self, csv_file: str, global_dim:int=50, device="cpu"):
        self.raw_data = self.read_data(csv_file) 
        self.global_dim = global_dim
        self.device = device


  
    def molecule_graph_to_tensor(self, smiles:str):


        mol_obj = Molecule(smiles)
        # Node features: atomic numbers.
        nodes = torch.tensor(mol_obj.nodes, dtype=torch.long, device=self.device)

        # Edge indices: convert list of tuples to a tensor [2, E]. (Note: Molecule stores (target, source))
        adjacency_list = torch.tensor(mol_obj.adjacency_list, dtype=torch.long, device=self.device).t().contiguous()

        # Edge features: bond types.
        edges = torch.tensor(mol_obj.edges, dtype=torch.long, device=self.device)

        # Initialize a learned global context vector for the graph (to be updated via message passing).
        global_context_vector = torch.zeros((1, self.global_dim), dtype=torch.float, device=self.device)  # Shape: [1, global_dim]
         

        return nodes, adjacency_list,  edges, global_context_vector 


    @staticmethod
    def read_data(path:str) -> dict:

        data = pd.read_csv(path, delimiter=",")

        molecules = {"smiles":data["IsomericSMILES"].to_numpy(),
                     "fruity":data["fruity"].to_numpy()}
        return molecules

    def __len__(self):
        return len(self.raw_data["smiles"])

    def __getitem__(self, index):

        smiles = self.raw_data[index]
        label = self.raw_data[index]

        nodes, adjacency_list, edges, global_context_vector = self.molecule_graph_to_tensor(smiles)

        sample = {
                "nodes": nodes,
                "edges": edges,
                "adjacency_list": adjacency_list,
                "label": label,
        }

        return sample


