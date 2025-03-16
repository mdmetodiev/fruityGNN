import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_mean
from utils.molecules import read_data

# Assume Molecule class is defined as in your description and that you have
# bond_type_mapping available to encode bond types.
# from your_molecule_module import Molecule, bond_type_mapping

# ---------------------------
# 1. Build a PyG dataset from your molecules
# ---------------------------
class MoleculeDataset(InMemoryDataset):
    def __init__(self, smiles_list, labels, transform=None, pre_transform=None):
        self.smiles_list = smiles_list
        self.labels = labels
        super(MoleculeDataset, self).__init__('.', transform, pre_transform)
        self.data, self.slices = self.process_data()

    def process_data(self):
        data_list = []
        for smiles, label in zip(self.smiles_list, self.labels):
            # Create a molecule graph using your Molecule class.
            mol = Molecule(smiles=smiles)
            # Node features: here we simply use the atomic number as a 1D feature.
            x = torch.tensor(mol.nodes, dtype=torch.float).view(-1, 1)
            # Edge index: convert adjacency list (list of (i,j) tuples) to tensor.
            if len(mol.adjacency_list) > 0:
                # edge_index should be shape [2, num_edges]
                edge_index = torch.tensor(list(zip(*mol.adjacency_list)), dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            # Edge features: use the bond type (as a 1D feature)
            edge_attr = torch.tensor(mol.edges, dtype=torch.float).view(-1, 1)
            # Global context: molecular weight and logP (as a 2D feature)
            global_context = torch.tensor(mol.global_context_vector, dtype=torch.float).view(1, -1)
            # Label: binary outcome (smells fruity or not)
            y = torch.tensor([label], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                        global_context=global_context, y=y)
            data_list.append(data)
        return self.collate(data_list)

# ---------------------------
# 2. Define a custom GNN layer with separate message passing
# ---------------------------
class GNNLayer(nn.Module):
    def __init__(self, hidden_dim, num_node_mp=20, num_edge_mp=20):
        """
        hidden_dim: common hidden dimension for nodes, edges and global context.
        num_node_mp: number of iterations for node message passing.
        num_edge_mp: number of iterations for edge message passing.
        """
        super(GNNLayer, self).__init__()
        self.num_node_mp = num_node_mp
        self.num_edge_mp = num_edge_mp
        
        # MLP for node message: takes concatenated source node and edge features.
        self.node_message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # MLP to update node features after aggregation.
        self.node_update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # MLP to update edge features given the features of the two connected nodes.
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # MLP to update the global context using aggregated node and edge info.
        self.global_update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr, global_context, batch):
        # ---- Node Message Passing ----
        for _ in range(self.num_node_mp):
            # edge_index: [2, num_edges] with row=source, col=target.
            row, col = edge_index
            # Compute messages from each edge:
            # For each edge, concatenate the source node feature and its edge feature.
            messages = self.node_message_mlp(torch.cat([x[row], edge_attr], dim=1))
            # Aggregate messages for each target node (mean aggregation):
            agg_messages = scatter_mean(messages, col, dim=0, dim_size=x.size(0))
            # Update node features: concatenate current node feature with aggregated message.
            x = self.node_update_mlp(torch.cat([x, agg_messages], dim=1))
        
        # ---- Edge Message Passing ----
        for _ in range(self.num_edge_mp):
            row, col = edge_index
            # Update edge attributes based on features of both incident nodes and current edge feature.
            edge_attr = self.edge_update_mlp(torch.cat([x[row], x[col], edge_attr], dim=1))
        
        # ---- Global Context Update ----
        # Pool node features per graph.
        node_pool = global_mean_pool(x, batch)  # shape: [num_graphs, hidden_dim]
        # For edge features, we assign each edge the batch index of its target node.
        edge_pool = global_mean_pool(edge_attr, batch[col])
        # Concatenate global context with pooled node and edge information.
        global_context = self.global_update_mlp(torch.cat([global_context, node_pool, edge_pool], dim=1))
        return x, edge_attr, global_context

# ---------------------------
# 3. Build the overall GNN model
# ---------------------------
class FruityGNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, global_in_dim, hidden_dim=64, 
                 num_layers=3, num_node_mp=20, num_edge_mp=20):
        """
        node_in_dim: dimensionality of raw node features (e.g. 1 for atomic number)
        edge_in_dim: dimensionality of raw edge features (e.g. 1 for bond type)
        global_in_dim: dimensionality of global context (2: mol weight and logP)
        hidden_dim: embedding size for all representations
        num_layers: number of overall GNN layers (each with message passing)
        num_node_mp: number of iterations for node message passing per GNN layer
        num_edge_mp: number of iterations for edge message passing per GNN layer
        """
        super(FruityGNN, self).__init__()
        # Initial embeddings for nodes, edges, and global context.
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim)
        self.global_embed = nn.Linear(global_in_dim, hidden_dim)
        
        # Stack several GNN layers.
        self.layers = nn.ModuleList([
            GNNLayer(hidden_dim, num_node_mp, num_edge_mp) for _ in range(num_layers)
        ])
        
        # Final classifier: combine pooled node features with global context.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, data):
        # Unpack data from PyG Data object.
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # global_context is stored per graph.
        global_context = data.global_context  # shape: [num_graphs, global_in_dim]
        
        # Embed raw features.
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        global_context = self.global_embed(global_context)
        
        # Propagate through the stacked GNN layers.
        for layer in self.layers:
            x, edge_attr, global_context = layer(x, edge_index, edge_attr, global_context, batch)
        
        # Global pooling on node features.
        node_pool = global_mean_pool(x, batch)  # shape: [num_graphs, hidden_dim]
        # Concatenate with global context.
        out = torch.cat([node_pool, global_context], dim=1)
        logits = self.classifier(out)
        # Use sigmoid activation for binary classification.
        return torch.sigmoid(logits).view(-1)

# ---------------------------
# 4. Training and Evaluation
# ---------------------------
# (Assume smiles_list and labels are provided lists.)
# For demonstration, here are dummy lists:

data = read_data("../data/curated_leffingwell.csv")
smiles_list = data["smiles"][::10]
labels = data["fruity"][::10]

# Create dataset and split into training and test sets.
dataset = MoleculeDataset(smiles_list, labels)
# For simplicity, we take a random 80/20 split.
num_total = len(dataset)
num_train = int(0.8 * num_total)
num_test = num_total - num_train
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])

# Create data loaders.
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Set up device, model, optimizer, and loss.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FruityGNN(node_in_dim=1, edge_in_dim=1, global_in_dim=2,
                   hidden_dim=64, num_layers=3, num_node_mp=20, num_edge_mp=20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training loop.
num_epochs = 10  # Increase as needed
for epoch in range(num_epochs):
    model.train()
    epoch_losses = []
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        preds = model(batch)
        loss = criterion(preds, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    avg_loss = np.mean(epoch_losses)
    
    # Evaluation on the test set.
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            all_preds.append(out.cpu())
            all_labels.append(batch.y.view(-1).cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    preds_binary = (all_preds >= 0.5).astype(int)
    accuracy = np.mean(preds_binary == all_labels)
    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}, AUC = {auc:.4f}")
