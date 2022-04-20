import torch

NUM_SAMPLES = 10000
NUM_NODES = 10
X = torch.zeros(NUM_SAMPLES*6, NUM_NODES, NUM_NODES)
Y = torch.zeros(NUM_SAMPLES*6, NUM_NODES)
for i in range(0, NUM_SAMPLES, 6):
    adj_mat = torch.randint(0, 2, (NUM_NODES, NUM_NODES))
    for j in range(NUM_NODES):
        adj_mat[j, j] = 0
    for j in range(NUM_NODES):
        for k in range(j+1, NUM_NODES):
            adj_mat[j, k] = adj_mat[k, j]
    X[i,:,:] = adj_mat
    for j in range(1, 6):
        perm = torch.randperm(NUM_NODES)
        X[i+j,:,:] = adj_mat[perm, perm]
Y = torch.reshape(torch.sum(X, dim=2), (NUM_SAMPLES*6, NUM_NODES))
torch.save(X, "../dataset/nodes.pt")
torch.save(Y, "../datset/labels.pt")
