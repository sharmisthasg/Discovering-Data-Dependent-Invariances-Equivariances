import torch
import numpy as np
import itertools
import pandas as pd

hps = [100, 10, 1, 1e-2, 1e-3, 1e-4, 1e-5]
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
class CustomModel(torch.nn.Module):
    def __init__(self, D, C):
        super(CustomModel, self).__init__()
        self.linear_layer_1 = torch.nn.Linear(D, D, bias=False)
        
    def forward(self, x):
        layer_1_output = self.linear_layer_1(x)
        layer_2_output = torch.sum(layer_1_output)
        return layer_2_output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#N = 100  # number of samples
D = 10  # input dimension
C = 1  # output dimension
data_path = "../../../Dataset/"
training_file = "Training.csv"
validation_file = "Validation.csv"
test_file = "Test.csv"
#X = torch.rand(N, D).to(device)  # (N, D)
X = torch.tensor(pd.read_csv(data_path + training_file, usecols=list(range(10))).to_numpy()).double().to(device)
#y = torch.sum(X, axis=-1).reshape(-1, C)  # (N, C)
y = torch.tensor(pd.read_csv(data_path + training_file, usecols=[10]).to_numpy()).reshape(-1, C).double().to(device) 
lr = 1e-2  # Learning rate

#model = torch.nn.Sequential(torch.nn.Linear(D, D, bias=False), torch.nn.ReLU(), torch.nn.Linear(D, C, bias=False), torch.nn.ReLU())  # model
model = CustomModel(D, C)
model.to(device)
model = model.double()
linear_layers = [layer for layer in model.children() if isinstance(layer, torch.nn.Linear)]
criterion = torch.nn.L1Loss(reduction='mean')  # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer
best_val_loss = float('inf')
best_hp = 1
model_path = "../../../models/one_hidden_layer/fixed_last/no_relu/"
torch.save(model.state_dict(), model_path + "untrained.pt")
for hp in hps:
    model.load_state_dict(torch.load(model_path + "untrained.pt"))
    model.train()
    best_training_loss = float('inf')
    for epoch in range(1000):
        y_pred = model(X)  # forward step
        loss = custom_loss_fn(criterion, y_pred, y, hp, linear_layers)  # compute loss
        loss.backward()  # backprop (compute gradients)
        optimizer.step()  # update weights (gradient descent step)
        optimizer.zero_grad()  # reset gradients
        if loss.item() < best_training_loss:
            best_training_loss = loss.item()
            torch.save(model.state_dict(), model_path + "best_training_" + str(hp) + ".pt")
    print("Best training loss for hp = " + str(hp) + " is " + str(best_training_loss))
    model.load_state_dict(torch.load(model_path + "best_training_" + str(hp) + ".pt"))
    model.eval()
    with torch.no_grad():
        print("Layer 1 parameters of the best model are: ", [x.data for x in model.parameters()][0])
        #print("Layer 2 parameters of the best model are: ", [x.data for x in model.parameters()][1])
        X_val = torch.tensor(pd.read_csv(data_path + validation_file, usecols=list(range(10))).to_numpy()).double().to(device)
        y_val = torch.tensor(pd.read_csv(data_path + validation_file, usecols=[10]).to_numpy()).reshape(-1, C).double().to(device)
        val_preds = model(X_val)
        eval_metric = criterion(val_preds, y_val).item()
        print("Validation evaluation loss on best model for hp = " + str(hp) + " is " + str(eval_metric))
        if eval_metric < best_val_loss:
            best_val_loss = eval_metric
            best_hp = hp
            torch.save(model.state_dict(), model_path + "best_validation.pt")
X_test = torch.tensor(pd.read_csv(data_path + test_file, usecols=list(range(10))).to_numpy()).double().to(device)
y_test = torch.tensor(pd.read_csv(data_path + test_file, usecols=[10]).to_numpy()).reshape(-1, C).double().to(device)
model.load_state_dict(torch.load(model_path + "best_validation.pt"))
print("Best layer 1 validation model parameters:", [x.data for x in model.parameters()][0])
#print("Best layer 2 validation model parameters:", [x.data for x in model.parameters()][1])
print("Best hyperparameter is:", best_hp)
model.eval()
with torch.no_grad():
    test_preds = model(X_test)
    test_eval_loss = criterion(test_preds, y_test).item()
    print("Test loss of best validation model is", test_eval_loss)
