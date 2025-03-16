import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from gnnfruity.utils.datareader import MoleculeGraphDataset 

from torch_scatter import scatter_add  # for sum aggregation in message passing
from sklearn.metrics import accuracy_score, roc_auc_score


# ---------------------------
# Custom GNN Layer Definition
# ---------------------------

class FruityGNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim):
        super(FruityGNNLayer, self).__init__()
        # Compute messages from a source node and its connecting edge.
        self.node_message_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        # Update a node by combining: its current state, aggregated messages, and the global context.
        self.node_update_mlp = nn.Sequential(
            nn.Linear(node_dim + node_dim + global_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        # Update an edge by combining its own state, source & target node features, and the global context.
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim + global_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim)
        )
        # Update the global context by aggregating node and edge information.
        self.global_update_mlp = nn.Sequential(
            nn.Linear(global_dim + node_dim + edge_dim, global_dim),
            nn.ReLU(),
            nn.Linear(global_dim, global_dim)
        )
        
    def forward(self, x, edge_index, edge_attr, global_context_vector, batch):
        # x: [N, node_dim] node embeddings.
        # edge_index: [2, E] connectivity.
        # edge_attr: [E, edge_dim] edge embeddings.
        # global_context_vector: [num_graphs, global_dim] global context for each graph.
        # batch: [N] mapping from nodes to their corresponding graph.
        
        source, target = edge_index  # source and target node indices per edge.
        
        # --- Node Message Passing ---
        # Compute messages from each source node conditioned on its edge.
        msg = self.node_message_mlp(torch.cat([x[source], edge_attr], dim=-1))
        # Sum-aggregate messages at each target node.
        agg_msg = scatter_add(msg, target, dim=0, dim_size=x.size(0))
        # Lookup the global context for each node based on its graph assignment.
        global_per_node = global_context_vector[batch]
        # Update node embeddings conditioned on its old state, aggregated messages, and global context.
        x_updated = self.node_update_mlp(torch.cat([x, agg_msg, global_per_node], dim=-1))
        
        # --- Edge Message Passing ---
        # For each edge, condition on its current state, the features of its source and target nodes,
        # and the corresponding global context (here taken from the source node’s graph).
        edge_global = global_context_vector[batch[source]]
        edge_updated = self.edge_update_mlp(torch.cat([edge_attr, x[source], x[target], edge_global], dim=-1))
        
        # --- Global Message Passing ---
        # Aggregate updated node embeddings per graph.
        agg_nodes = scatter_add(x_updated, batch, dim=0)
        # For edges, we assume each edge belongs to the same graph as its source node.
        edge_batch = batch[source]
        agg_edges = scatter_add(edge_updated, edge_batch, dim=0, dim_size=global_context_vector.size(0))
        # Concatenate the previous global context with aggregated node and edge messages.
        global_input = torch.cat([global_context_vector, agg_nodes, agg_edges], dim=-1)
        global_updated = self.global_update_mlp(global_input)
        
        return x_updated, edge_updated, global_updated

# ---------------------------
# FruityGNN Model Definition
# ---------------------------

class FruityGNN(nn.Module):
    def __init__(self, num_layers=4, node_dim=50, edge_dim=50, global_dim=50,
                 max_atomic_num=100, num_edge_types=5):
        """
        Parameters:
          num_layers: Number of sequential GNN layers (default: 3)
          node_dim: Embedding size for nodes (default: 20)
          edge_dim: Embedding size for edges (default: 20)
          global_dim: Embedding size for the global context vector (default: 20)
          max_atomic_num: Maximum atomic number (for node embedding lookup)
          num_edge_types: Number of distinct bond types (for edge embedding lookup)
        """
        super(FruityGNN, self).__init__()
        # Embedding layers for node atomic numbers and edge bond types.
        self.node_embedding = nn.Embedding(max_atomic_num + 1, node_dim)
        self.edge_embedding = nn.Embedding(num_edge_types, edge_dim)
        # Note: Unlike before, the global context is not based on external features.
        # It is instead a learned representation that is initialized per graph.
        
        # Stack the customizable GNN layers.
        self.layers = nn.ModuleList([
            FruityGNNLayer(node_dim, edge_dim, global_dim) for _ in range(num_layers)
        ])
        
        # Final classifier: the global context (graph-level representation) is used for prediction.
        self.classifier = nn.Sequential(
            nn.Linear(global_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, data):
        """
        data: a PyG Data object with attributes:
              - x: [N] long tensor (atomic numbers)
              - edge_index: [2, E] long tensor (edge connectivity)
              - edge_attr: [E] long tensor (bond types)
              - global_context_vector: [num_graphs, global_dim] tensor (initialized global context)
              - batch: [N] long tensor mapping nodes to graphs
              - y: [num_graphs] target labels (optional)
        """
        # Embed node and edge features.
        x = self.node_embedding(data.x)           # shape: [N, node_dim]
        edge_attr = self.edge_embedding(data.edge_attr)  # shape: [E, edge_dim]
        # Use the provided global context; this is a learned vector for each graph.
        global_context_vector = data.global_context_vector             # shape: [num_graphs, global_dim]
        
        # Sequential message passing through the layers.
        for layer in self.layers:
            x, edge_attr, global_context_vector = layer(x, data.edge_index, edge_attr, global_context_vector, data.batch)
            
        # Final prediction from the updated global context.
        out = self.classifier(global_context_vector).squeeze(-1)  # shape: [num_graphs]
        return out

# ---------------------------
# Example Dataset Preparation
# ---------------------------
# Assume you have:
#   smiles_list: list of SMILES strings
#   labels: list of binary outcomes (0 or 1)
# and that you have defined a bond_type_mapping and the Molecule class as given.
# For demonstration, we assume these exist.
#
# Example:



TRAIN = True 

if TRAIN == True:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"using {device} as device")
    dataset = MoleculeGraphDataset("data/curated_leffingwell.csv", device=device) 
    train_data, test_data = random_split(dataset, [0.2, 0.8])
    

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    # ---------------------------
    # Training Loop
    # ---------------------------
    model = FruityGNN(num_layers=4, node_dim=50, edge_dim=50, global_dim=50,
                      max_atomic_num=100, num_edge_types=5).to(device)
    criterion = nn.BCELoss()  # since our classifier already applies sigmoid
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 100# adjust epochs as needed

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y.to(device).view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.num_graphs
            
            preds = (out > 0.5).float().detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(data.y.cpu().numpy())
            
        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)
        train_auc = roc_auc_score(all_labels, all_preds)
        
        # Evaluate on test set
        model.eval()
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                preds = (out > 0.5).float().detach().cpu().numpy()
                test_preds.extend(preds)
                test_labels.extend(data.y.cpu().numpy())
        test_acc = accuracy_score(test_labels, test_preds)
        test_auc = roc_auc_score(test_labels, test_preds)
        
        print(f"Epoch {epoch:02d}: Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f} | "
              f"Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

    # ---------------------------
    # Save the Model
    # ---------------------------
    torch.save(model.state_dict(), "fruitygnn_model.pth")
    print("Model saved to fruitygnn_model.pth")
