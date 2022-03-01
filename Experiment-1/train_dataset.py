import torch
import numpy as np
import itertools
import pandas as pd

hps = [10, 1, 1e-2, 1e-3, 1e-4]
def f(a):
    return abs(a[0] - a[1])
def custom_loss_fn(criterion, model, y_pred, y, hp):
    all_pairs = list(itertools.combinations([x.data for x in model.parameters()][0], 2))
    return criterion(y_pred, y) + hp*sum(list(map(f, all_pairs)))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#N = 100  # number of samples
D = 10  # input dimension
C = 1  # output dimension

#X = torch.rand(N, D).to(device)  # (N, D)
X = torch.tensor(pd.read_csv("Dataset/Training.csv", usecols=list(range(10))).to_numpy()).double().to(device)
#y = torch.sum(X, axis=-1).reshape(-1, C)  # (N, C)
y = torch.tensor(pd.read_csv("Dataset/Training.csv", usecols=[10]).to_numpy()).reshape(-1, C).double().to(device) 
lr = 1e-2  # Learning rate

model = torch.nn.Sequential(torch.nn.Linear(D, C, bias=False), torch.nn.ReLU())  # model
model.to(device)
model = model.double()
criterion = torch.nn.L1Loss(reduction='mean')  # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer
best_val_loss = float('inf')
best_hp = 1
torch.save(model.state_dict(), "models/untrained.pt")
for hp in hps:
    model.load_state_dict(torch.load("models/untrained.pt"))
    model.train()
    best_training_loss = float('inf')
    for epoch in range(1000):
        y_pred = model(X)  # forward step
        loss = custom_loss_fn(criterion, model, y_pred, y, hp)  # compute loss
        loss.backward()  # backprop (compute gradients)
        optimizer.step()  # update weights (gradient descent step)
        optimizer.zero_grad()  # reset gradients
        if loss.item() < best_training_loss:
            best_training_loss = loss.item()
            torch.save(model.state_dict(), "models/best_training_" + str(hp) + ".pt")
    print("Best training loss for hp = " + str(hp) + " is " + str(best_training_loss))
    model.load_state_dict(torch.load("models/best_training_" + str(hp) + ".pt"))
    model.eval()
    with torch.no_grad():
        print("Parameters of the best model are: ", [x.data for x in model.parameters()][0])
        X_val = torch.tensor(pd.read_csv("Dataset/Validation.csv", usecols=list(range(10))).to_numpy()).double().to(device)
        y_val = torch.tensor(pd.read_csv("Dataset/Validation.csv", usecols=[10]).to_numpy()).reshape(-1, C).double().to(device)
        val_preds = model(X_val)
        eval_metric = criterion(val_preds, y_val).item()
        print("Validation evaluation loss on best model for hp = " + str(hp) + " is " + str(eval_metric))
        if eval_metric < best_val_loss:
            best_val_loss = eval_metric
            best_hp = hp
            torch.save(model.state_dict(), "models/best_validation.pt")
X_test = torch.tensor(pd.read_csv("Dataset/Test.csv", usecols=list(range(10))).to_numpy()).double().to(device)
y_test = torch.tensor(pd.read_csv("Dataset/Test.csv", usecols=[10]).to_numpy()).reshape(-1, C).double().to(device)
model.load_state_dict(torch.load("models/best_validation.pt"))
print("Best validation model parameters:", [x.data for x in model.parameters()][0])
print("Best hyperparameter is:", best_hp)
model.eval()
with torch.no_grad():
    test_preds = model(X_test)
    test_eval_loss = criterion(test_preds, y_test).item()
    print("Test loss of best validation model is", test_eval_loss)
