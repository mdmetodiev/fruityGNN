import pandas as pd
from gnnfruity.utils.molecule import Molecule
from torch_geometric.data import Dataset, Data
import torch

class MoleculeGraphDataset(Dataset):
    def __init__(self, csv_file: str, global_dim: int = 50, device="cpu"):
        super().__init__()
        self.raw_data = self.read_data(csv_file)
        self.global_dim = global_dim
        self.device = device

    @staticmethod
    def read_data(path: str) -> dict:
        data = pd.read_csv(path, delimiter=",")
        return {
            "smiles": data["IsomericSMILES"].to_numpy(),
            "fruity": data["fruity"].to_numpy()
        }

    def len(self):
        return len(self.raw_data["smiles"])

    def get(self, idx: int) -> Data:
        smiles = self.raw_data["smiles"][idx]
        label = self.raw_data["fruity"][idx]

        # Convert your molecule to PyG Data format
        mol_obj = Molecule(smiles)

        # Node features (atomic numbers)
        x = torch.tensor(mol_obj.nodes, dtype=torch.long, device=self.device)

        # Edge indices (adjacency list in COO format)
        edge_index = torch.tensor(mol_obj.adjacency_list, dtype=torch.long, device=self.device).t().contiguous()

        # Edge features (bond types)
        edge_attr = torch.tensor(mol_obj.edges, dtype=torch.long, device=self.device)
        
        # Initialize a learned global context vector for the graph (to be updated via message passing).
        global_context_vector = torch.zeros((1, self.global_dim), dtype=torch.float, device=self.device)  # Shape: [1, global_dim]

        # Label
        y = torch.tensor([label], dtype=torch.float)

        # Create PyG Data object
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_context_vector=global_context_vector,
            y=y
        )
