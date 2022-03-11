import torch
import numpy as np
import itertools

hp = 1
def f(a):
    return hp*abs(a[0] - a[1])
def custom_loss_fn(criterion, model, y_pred, y):
    all_pairs = list(itertools.combinations([x.data for x in model.parameters()][0], 2))
    return criterion(y_pred, y) + sum(list(map(f, all_pairs)))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N = 100  # number of samples
D = 10  # input dimension
C = 1  # output dimension

X = torch.rand(N, D).to(device)  # (N, D)
y = torch.sum(X, axis=-1).reshape(-1, C)  # (N, C)

lr = 1e-2  # Learning rate

model = torch.nn.Sequential(torch.nn.Linear(D, C, bias=False), torch.nn.ReLU())  # model
model.to(device)

criterion = torch.nn.MSELoss()  # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer

for epoch in range(1000):
    y_pred = model(X)  # forward step
    loss = custom_loss_fn(criterion, model, y_pred, y)  # compute loss
    loss.backward()  # backprop (compute gradients)
    optimizer.step()  # update weights (gradient descent step)
    optimizer.zero_grad()  # reset gradients
    if epoch % 50 == 0:
        print(f"[EPOCH]: {epoch}, [LOSS]: {loss.item():.6f}")
        print("Model parameters are: ", [x.data for x in  model.parameters()][0])
