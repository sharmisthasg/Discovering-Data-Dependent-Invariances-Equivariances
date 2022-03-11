import torch
import numpy as np
import itertools
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
D = 10  # input dimension
C = 1  # output dimension
X = torch.tensor(pd.read_csv("Dataset/Training.csv", usecols=list(range(10))).to_numpy()).double().to(device)
y = torch.tensor(pd.read_csv("Dataset/Training.csv", usecols=[10]).to_numpy()).reshape(-1, C).double().to(device) 
lr = 1e-2  # Learning rate

model = torch.nn.Sequential(torch.nn.Linear(D, C, bias=False), torch.nn.ReLU())  # model
model.to(device)
model = model.double()
criterion = torch.nn.L1Loss(reduction='mean')  # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer
model.train()
best_training_loss = float('inf')
for epoch in range(1000):
    y_pred = model(X)  # forward step
    loss = criterion(y_pred, y)  # compute loss
    loss.backward()  # backprop (compute gradients)
    optimizer.step()  # update weights (gradient descent step)
    optimizer.zero_grad()  # reset gradients
    if loss.item() < best_training_loss:
        best_training_loss = loss.item()
        torch.save(model.state_dict(), "models/best_training_no_reg.pt")
print("Best training loss for is " + str(best_training_loss))
model.load_state_dict(torch.load("models/best_training_no_reg.pt"))
model.eval()
X_test = torch.tensor(pd.read_csv("Dataset/Test.csv", usecols=list(range(10))).to_numpy()).double().to(device)
y_test = torch.tensor(pd.read_csv("Dataset/Test.csv", usecols=[10]).to_numpy()).reshape(-1, C).double().to(device)
print("Best training model parameters:", [x.data for x in model.parameters()][0])
with torch.no_grad():
    test_preds = model(X_test)
    test_eval_loss = criterion(test_preds, y_test).item()
    print("Test loss of best training model is", test_eval_loss)
