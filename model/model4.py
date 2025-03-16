import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from gnnfruity.utils.datareader import MoleculeGraphDataset 


from sklearn.metrics import accuracy_score, roc_auc_score

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

class FruityGNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, global_dim):
        super().__init__(aggr='add', flow='source_to_target')
        
        # Node message network
        self.node_msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        # Node update network (now expects x + aggr + global)
        self.node_update_mlp = nn.Sequential(
            nn.Linear(2*node_dim + global_dim, node_dim),  # 50+50+50=150 → 50
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        # Edge update network
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(2*node_dim + edge_dim + global_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim)
        )
        
        # Global update network
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim + node_dim + edge_dim, global_dim),
            nn.ReLU(),
            nn.Linear(global_dim, global_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # 1. Node Message Passing
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)  # [N, node_dim]
        
        # 2. Node Update: Include global context
        global_per_node = u[batch]  # [N, global_dim]
        x_updated = self.node_update_mlp(
            torch.cat([x, aggr_out, global_per_node], dim=-1)  # [N, 150]
        )
        
        # 3. Edge Update
        row, col = edge_index
        edge_global = u[batch[row]]  # [E, global_dim]
        edge_attr_updated = self.edge_update_mlp(
            torch.cat([x_updated[row], x_updated[col], edge_attr, edge_global], dim=-1)
        )
        
        # 4. Global Update
        node_agg = scatter(x_updated, batch, dim=0, reduce='sum')  # [B, node_dim]
        edge_agg = scatter(edge_attr_updated, batch[row], dim=0, reduce='sum')  # [B, edge_dim]
        u_updated = self.global_mlp(torch.cat([u, node_agg, edge_agg], dim=-1))
        
        return x_updated, edge_attr_updated, u_updated

    def message(self, x_j, edge_attr):
        return self.node_msg_mlp(torch.cat([x_j, edge_attr], dim=-1))

    def update(self, aggr_out):
        return aggr_out  # Just pass-through, actual update done in forward

class FruityGNN(nn.Module):
    def __init__(self, num_layers=4, node_dim=50, edge_dim=50, global_dim=50,
                 max_atomic_num=100, num_edge_types=5):
        super().__init__()
        self.node_emb = nn.Embedding(max_atomic_num+1, node_dim)
        self.edge_emb = nn.Embedding(num_edge_types, edge_dim)
        self.global_init = nn.Parameter(torch.randn(1, global_dim))
        self.layers = nn.ModuleList([
            FruityGNNLayer(node_dim, edge_dim, global_dim) for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(global_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x = self.node_emb(data.x)
        edge_attr = self.edge_emb(data.edge_attr)
        u = self.global_init.repeat(data.num_graphs, 1)
        
        for layer in self.layers:
            x, edge_attr, u = layer(x, data.edge_index, edge_attr, u, data.batch)
            
        return self.classifier(u).squeeze(-1)


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
    torch.save(model.state_dict(), "fruitygnn_model2.pth")
    print("Model saved to fruitygnn_model2.pth")
