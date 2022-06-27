import torch
import numpy as np
import itertools
import pandas as pd
from sklearn.metrics import accuracy_score

hps = [100, 10, 1, 0, 1e-2, 1e-3, 1e-4, 1e-5]
def accuracy(a, b):
    return (a==b).sum().item()/torch.numel(a)
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
        self.linear_layer_2 = torch.nn.Linear(D, D, bias=False)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        layer_1_output = self.tanh(self.linear_layer_1(x))
        layer_2_output = self.linear_layer_2(x)
        layer_3_output = torch.nn.functional.softmax(layer_2_output, dim=1)
        return layer_3_output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N = 60000 # number of samples
D = 10  # input dimension
C = 10  # output dimension

X = torch.load("../dataset/nodes.pt").to(device).float()
Y = torch.load("../dataset/labels.pt").to(device).float()
lr = 1e-2  # Learning rate

X_train = X[:int(0.8*X.shape[0]), :, :].float()
Y_train = Y[:int(0.8*Y.shape[0]), :].long()
X_val = X[int(0.8*X.shape[0]):int(0.9*X.shape[0]), :, :].float()
Y_val = Y[int(0.8*Y.shape[0]):int(0.9*Y.shape[0]), :].long()
X_test = X[int(0.9*X.shape[0]):, :, :].float()
Y_test = Y[int(0.9*Y.shape[0]):, :].long()
model = CustomModel(D, C)
model.to(device)
model = model.float()
linear_layers = [layer for layer in model.children() if isinstance(layer, torch.nn.Linear)]
criterion = torch.nn.CrossEntropyLoss(reduction='mean')  # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer
best_val_acc = 0
best_hp = 1
torch.save(model.state_dict(), "../models/one_hidden_layer/fixed_last/untrained_nn_classification.pt")
for hp in hps:
    model.load_state_dict(torch.load("../models/one_hidden_layer/fixed_last/untrained_nn_classification.pt"))
    model.train()
    best_training_loss = float('inf')
    for epoch in range(1000):
        y_pred = model(X_train)  # forward step
        loss = custom_loss_fn(criterion, y_pred, Y_train, hp, linear_layers)  # compute loss
        loss.backward()  # backprop (compute gradients)
        optimizer.step()  # update weights (gradient descent step)
        optimizer.zero_grad()  # reset gradients
        if loss.item() < best_training_loss:
            best_training_loss = loss.item()
            torch.save(model.state_dict(), "../models/one_hidden_layer/fixed_last/nn_classification_best_training_" + str(hp) + ".pt")
    print("Best training loss for hp = " + str(hp) + " is " + str(best_training_loss))
    model.load_state_dict(torch.load("../models/one_hidden_layer/fixed_last/nn_classification_best_training_" + str(hp) + ".pt"))
    model.eval()
    with torch.no_grad():
        print("Layer 1 parameters of the best model are: ", [x.data for x in model.parameters()][0])
        val_preds = model(X_val)
        #eval_metric = accuracy_score(Y_val.detach().cpu().numpy(), torch.argmax(val_preds, dim=1).detach().cpu().numpy())
        eval_metric = accuracy(Y_val, torch.argmax(val_preds, dim=1))
        print("Validation evaluation accuracy on best model for hp = " + str(hp) + " is " + str(eval_metric))
        if eval_metric > best_val_acc:
            best_val_acc = eval_metric
            best_hp = hp
            torch.save(model.state_dict(), "../models/one_hidden_layer/fixed_last/nn_classification_best_validation.pt")
model.load_state_dict(torch.load("../models/one_hidden_layer/fixed_last/nn_classification_best_validation.pt"))
print("Best layer 1 validation model parameters:", [x.data for x in model.parameters()][0])
print("Best hyperparameter is:", best_hp)
model.eval()
with torch.no_grad():
    test_preds = model(X_test)
    #test_eval_loss = accuracy_score(Y_test.detach().cpu().numpy(), torch.argmax(test_preds, dim=1).detach().cpu().numpy())
    test_eval_acc = accuracy(Y_test, torch.argmax(test_preds, dim=1))
    print("Test accuracy of best validation model is", test_eval_acc)
