import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging
import numpy as np
import warnings
warnings.filterwarnings('always')
import dataset_2D_generator
from collections import OrderedDict
from torch.nn import functional as F

class Model2D(torch.nn.Module):
    def __init__(self):
        super(Model2D, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 5)
        
    def forward(self, x):
        x = x.reshape(x.size(dim=0), 1, x.size(dim=1), x.size(dim=2))
        x = self.conv1(x)
        x = torch.tanh(x)
        x,_ = torch.max(x, dim=-1)
        x,_ = torch.max(x, dim=-1)
        x = torch.nn.functional.softmax(x, dim=1)
        return x

def load_dataset():
    samples, targets = dataset_2D_generator.main()
    shuffled_indices = np.random.permutation(samples.shape[0])
    shuffled_samples = samples[shuffled_indices]
    shuffled_targets = targets[shuffled_indices]

    train_size = int(0.7 * len(shuffled_samples))
    validation_size = train_size + int(0.2 * len(shuffled_samples))

    X_train, y_train = shuffled_samples[:train_size], shuffled_targets[:train_size]
    X_validation, y_validation = shuffled_samples[train_size:validation_size], shuffled_targets[train_size:validation_size]
    X_test, y_test = shuffled_samples[validation_size:], shuffled_targets[validation_size:]

    return {"train": dict(X=torch.Tensor(X_train), y=torch.Tensor(y_train)),
            "test": dict(X=torch.Tensor(X_test), y=torch.Tensor(y_test)),
            "validation": dict(X=torch.Tensor(X_validation), y=torch.Tensor(y_validation))}


def train(model,criterion,optimizer,X,y):
    model.train()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()  # backprop (compute gradients)
    optimizer.step()  # update weights (gradient descent step)
    optimizer.zero_grad()  # reset gradients
    return loss


def test(model,criterion,X,y,validation):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)  # forward step
        if validation:
            eval_metric = criterion(y_pred, y)  # compute loss
            return eval_metric
        else:
            #print("y_pred", y_pred)
            return torch.argmax(y_pred,dim=1)

def main():
    torch.manual_seed(696)
    data = load_dataset()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train, y_train = data['train']['X'].to(device), data['train']['y'].type(torch.LongTensor).to(device)
    X_validation, y_validation = data['validation']['X'].to(device), data['validation']['y'].type(torch.LongTensor).to(device)
    X_test, y_test = data['test']['X'].to(device), data['test']['y'].type(torch.LongTensor).to(device)
    lr = 5e-2  # Learning rate

    model = Model2D().to(device)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')  # loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer

    # training the model for 1000 epochs
    optimum_training_loss = float('inf')
    optimum_validation_loss = float('inf')

    for epoch in range(5):
        training_loss = 0
        validation_loss = 0
        training_loss = train(model,criterion,optimizer,X_train,y_train).data.cpu()
        validation_loss = test(model,criterion,X_validation,y_validation,validation=True).detach().data.cpu()
        print(f'training loss: {training_loss}, validation loss:{validation_loss}')
        if training_loss < optimum_training_loss and validation_loss < optimum_validation_loss:
            optimum_validation_loss = validation_loss
            optimum_training_loss = training_loss
            torch.save(model.state_dict(), "../models/conv2d_cnn.pt")

    model.load_state_dict(torch.load("../models/conv2d_cnn.pt"))
    accuracy = 0
    model.eval()
    pred = test(model, criterion, X_test, y_test, validation=False)
    accuracy = torch.eq(pred, y_test).long().sum()/pred.size(dim=0)
    print(f'Test accuracy: {accuracy}')

if __name__ == '__main__':
    main()
