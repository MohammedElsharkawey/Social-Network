#Mohammed Yasser El.Sharkawey
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt

# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# High-accuracy GCN model
class BetterGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Hidden dimension increased from 4 → 16 (boosts accuracy)
        self.conv1 = GCNConv(data.num_node_features, 16)

        # Output layer
        self.conv2 = GCNConv(16, dataset.num_classes)

        # Dropout to prevent overfitting
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Layer 1 + ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Dropout layer (improves generalization)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# Build model
model = BetterGCN()

# Adam optimizer with weight decay (prevents overfitting)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# Training loop
for epoch in range(300):  # Train longer for higher accuracy
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Evaluate accuracy on test nodes
model.eval()
pred = model(data).argmax(dim=1)
acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
print(f'Improved GCN Test Accuracy: {acc:.2f}')