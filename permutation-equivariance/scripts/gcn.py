import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np

hps = [100, 10, 1, 0, 1e-2, 1e-3, 1e-4, 1e-5]

def f(a):
    return abs(a[0] - a[1])

def custom_loss_fn(criterion, y_pred, y, hp, linear_layers):
    layers = []
    for layer in linear_layers:
        all_pairs = list(itertools.combinations([x.data for x in layer.parameters()][0], 2))
        layers.append(all_pairs)
    reg_sum = 0
    for index in range(len(linear_layers)):
        try:
            reg_sum += hp * sum(sum(list(map(f, layers[index]))))
        except:
            reg_sum += hp * sum(list(map(f, layers[index])))
    return criterion(y_pred, y) + reg_sum

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(10, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, C, normalize=False))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            print(x.size())
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

# class GNN(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super().__init__()
#         torch.manual_seed(696)
#         self.conv1 = GCNConv(10, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, C)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         print("x in model conv1", x)
#         x = x.relu()
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index)
#         print("x in model conv2", x)
#         return x

# class GNN(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super().__init__()
#         self.conv1 = GCNConv(1, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, C)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N = 60000 # number of samples
D = 10  # input dimension
C = 10  # output dimension

X = torch.load("../dataset/nodes.pt").to(device).float()
Y = torch.load("../dataset/labels.pt").to(device).float()
lr = 1e-2  # Learning rate

data_arr = []
for i in range(N):
    adj_mat = X[i, :, :]
    edge_index = torch.where(adj_mat > 0)
    edge_index = torch.stack(edge_index)
    data = Data(x = torch.tensor(list(range(0, 10))), edge_index = edge_index, y = Y[i])
    data_arr.append(data)
print(len(data_arr))

X_train = data_arr[:int(0.8*X.shape[0])]
Y_train = Y[:int(0.8*Y.shape[0])]
X_val = data_arr[int(0.8*X.shape[0]):int(0.9*X.shape[0])]
Y_val = Y[int(0.8*Y.shape[0]):int(0.9*Y.shape[0])]
X_test = data_arr[int(0.9*X.shape[0]):]
Y_test = Y[int(0.9*Y.shape[0]):]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(hidden_channels=40)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    print("Epoch", epoch)
    model.train()
    for i in range(len(X_train)):
        x, edge_index, y = X_train[i].x, X_train[i].edge_index, X_train[i].y
        optimizer.zero_grad()
        x = x.type(torch.LongTensor)
        out = model(x, edge_index)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

correct = 0
total = 0
model.eval()
for i in range(len(X_test)):
    x, edge_index, y = X_test[i].x, X_test[i].edge_index, X_test[i].y
    out = model(x, edge_index)
    pred = out.argmax(dim=1)
    correct += (pred == y).sum()
    total += len(y)
print("Accuracy: ", correct/total)

# criterion = torch.nn.L1Loss(reduction='mean')  # loss function
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer

# best_val_loss = float('inf')
# best_hp = 1
# torch.save(model.state_dict(), "../models/gnn/vanilla_gnn.pt")
#
# for hp in hps:
#     model.load_state_dict(torch.load("../models/gnn/vanilla_gnn.pt"))
#     model.train()
#     best_training_loss = float('inf')
#     for epoch in range(100):
#         y_pred = model(X_train)  # forward step
#         loss = custom_loss_fn(criterion, y_pred, Y_train, hp, linear_layers)  # compute loss
#         loss.backward()  # backprop (compute gradients)
#         optimizer.step()  # update weights (gradient descent step)
#         optimizer.zero_grad()  # reset gradients
#         if loss.item() < best_training_loss:
#             best_training_loss = loss.item()
#             torch.save(model.state_dict(), "../models/gnn/vanilla_gnn_best_training_" + str(hp) + ".pt")
#     print("Best training loss for hp = " + str(hp) + " is " + str(best_training_loss))
#     model.load_state_dict(torch.load("../models/gnn/vanilla_gnn_best_training_" + str(hp) + ".pt"))
#     model.eval()
#     with torch.no_grad():
#         print("Layer 1 parameters of the best model are: ", [x.data for x in model.parameters()][0])
#         val_preds = model(X_val)
#         eval_metric = criterion(val_preds, Y_val).item()
#         print("Validation evaluation loss on best model for hp = " + str(hp) + " is " + str(eval_metric))
#         if eval_metric < best_val_loss:
#             best_val_loss = eval_metric
#             best_hp = hp
#             torch.save(model.state_dict(), "../models/gnn/vanilla_gnn_best_validation.pt")
# model.load_state_dict(torch.load("../models/gnn/vanilla_gnn_best_validation.pt"))
# print("Best layer 1 validation model parameters:", [x.data for x in model.parameters()][0])
# print("Best hyperparameter is:", best_hp)
# model.eval()
# with torch.no_grad():
#     test_preds = model(X_test)
#     test_eval_loss = criterion(test_preds, Y_test).item()
#     print("Test loss of best validation model is", test_eval_loss)