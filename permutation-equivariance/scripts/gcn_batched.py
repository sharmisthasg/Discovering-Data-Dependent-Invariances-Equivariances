import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 10)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)
N = 60000 # number of samples
D = 10  # input dimension
C = 10  # output dimension

X = torch.load("../dataset/nodes.pt").to(device).long()
Y = torch.load("../dataset/labels.pt").to(device).long()
lr = 1e-2  # Learning rate
print("Starting to prepare training data")
data_train = []
batch_size = 32
for i in range(int(0.8*N), batch_size):
    edge_index = torch.zeros((2, 0))
    for j in range(i, i+batch_size):
        adj_mat = X[j, :, :]
        temp = torch.nonzero(adj_mat)
        temp = torch.transpose(temp, 0, 1) + (j-i)*10
        edge_index = torch.cat((edge_index, temp), dim=1)
    data_train.append(Data(x = torch.ones(batch_size*D, 1), edge_index=edge_index.long(), y=Y[i:i+batch_size]))
print("Starting to prepare testing data")

data_test = []
for i in range(int(0.8*N), N, batch_size):
    edge_index = torch.zeros((2, 0))
    for j in range(i, i+batch_size):
        adj_mat = X[j, :, :]
        temp = torch.nonzero(adj_mat)
        temp = torch.transpose(temp, 0, 1) + (j-i)*10
        edge_index = torch.cat((edge_index, temp), dim=1)
    data_test.append(Data(x = torch.ones(batch_size*D, 1), edge_index=edge_index.long(), y=Y[i:i+batch_size]))

print("Created data array")
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
print("Training begins")

for epoch in range(10):
    print("Epoch", epoch)
    model.train()
    for data in data_train:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, torch.reshape(data.y, (data.y.shape[0]*D,)))
        loss.backward()
        optimizer.step()
torch.save(model, "gcn_model.pt")
print("testing begins")
model.eval()
correct = 0
total = 0
for data in data_test:
    pred = model(data).argmax(dim=1)
    correct += (pred == torch.reshape(data.y, (-1, 1))).sum().item()
    total += torch.numel(data.y)
acc = correct/total
print(f'Accuracy: {acc:.4f}')
