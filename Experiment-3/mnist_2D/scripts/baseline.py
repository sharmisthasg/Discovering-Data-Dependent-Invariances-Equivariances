import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging
import numpy as np
import warnings
warnings.filterwarnings('always')
from collections import OrderedDict
from torch.nn import functional as F
import sys
sys.path.append('../datasets')
import Load_dataset


class Baseline(torch.nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, 28)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x, _ = torch.max(x, dim=-1)
        x, _ = torch.max(x, dim=-1)
        return x


def train(model, criterion, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        prediction = model(data)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def test(model,criterion,loader,device,validation):
    model.eval()
    total_loss = 0
    accuracy = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            prediction = model(data)
            if validation:
                loss = criterion(prediction, target)
                total_loss += loss
            else:
                predictions = prediction.argmax(dim=1, keepdim=True)
                accuracy.append(accuracy_score(target.cpu(), predictions.cpu()))
    if validation:
        return total_loss
    else:
        return accuracy


def main():
    torch.manual_seed(696)
    train_loader, validation_loader, test_loader = Load_dataset.main()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lr = 5e-2  # Learning rate

    model = Baseline().to(device)

    criterion = torch.nn.CrossEntropyLoss()  # loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer

    # training the model for 1000 epochs
    optimum_training_loss = float('inf')
    optimum_validation_loss = float('inf')

    for epoch in range(40):
        training_loss = train(model,criterion,optimizer,train_loader,device)
        validation_loss = test(model,criterion,validation_loader,device,validation=True)
        print(f'training loss: {training_loss}, validation loss:{validation_loss}')
        if training_loss < optimum_training_loss and validation_loss < optimum_validation_loss:
            optimum_validation_loss = validation_loss
            optimum_training_loss = training_loss
            torch.save(model.state_dict(), "../models/2D_mnist_baseline" + ".pt")

    model.load_state_dict(torch.load("../models/2D_mnist_baseline" + ".pt"))
    accuracy = test(model,criterion,test_loader,device,validation=False)
    print(f'Test score: {sum(accuracy)/len(accuracy)}')


if __name__ == '__main__':
    main()