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


class Model2D(torch.nn.Module):
    def __init__(self):
        super(Model2D, self).__init__()
        weight1 = torch.randn(1, 1,    14 * 14,   14 * 14)
        self.weight1 = torch.nn.Parameter(weight1)
        weight2 = torch.randn(1, 1,    14 * 14,   14 * 14)
        self.weight2 = torch.nn.Parameter(weight2)
        weight3 = torch.randn(1, 1,    14 * 14,   14 * 14)
        self.weight3 = torch.nn.Parameter(weight3)
        weight4 = torch.randn(1, 1,    14 * 14,   14 * 14)
        self.weight4 = torch.nn.Parameter(weight4)
        weight5 = torch.randn(1, 1,    14 * 14,   14 * 14)
        self.weight5 = torch.nn.Parameter(weight5)
        weight6 = torch.randn(1, 1,    14 * 14,   14 * 14)
        self.weight6 = torch.nn.Parameter(weight6)
        weight7 = torch.randn(1, 1,    14 * 14,   14 * 14)
        self.weight7 = torch.nn.Parameter(weight7)
        weight8 = torch.randn(1, 1,    14 * 14,   14 * 14)
        self.weight8 = torch.nn.Parameter(weight8)
        weight9 = torch.randn(1, 1,    14 * 14,   14 * 14)
        self.weight9 = torch.nn.Parameter(weight9)
        weight10 = torch.randn(1, 1,   14 * 14,   14 * 14)
        self.weight10 = torch.nn.Parameter(weight10)

    def forward(self, x):
        output1 = torch.tanh((F.conv2d(self.weight1, x, stride=(28,28)))).transpose(0,1)
        output2 = torch.tanh((F.conv2d(self.weight2, x, stride=(28,28)))).transpose(0,1)
        output3 = torch.tanh((F.conv2d(self.weight3, x, stride=(28,28)))).transpose(0,1)
        output4 = torch.tanh((F.conv2d(self.weight4, x, stride=(28,28)))).transpose(0,1)
        output5 = torch.tanh((F.conv2d(self.weight5, x, stride=(28,28)))).transpose(0,1)
        output6 = torch.tanh((F.conv2d(self.weight6, x, stride=(28,28)))).transpose(0,1)
        output7 = torch.tanh((F.conv2d(self.weight7, x, stride=(28,28)))).transpose(0,1)
        output8 = torch.tanh((F.conv2d(self.weight8, x, stride=(28,28)))).transpose(0,1)
        output9 = torch.tanh((F.conv2d(self.weight9, x, stride=(28,28)))).transpose(0,1)
        output10 = torch.tanh((F.conv2d(self.weight10, x, stride=(28,28)))).transpose(0,1)
        output = torch.cat((output1, output2, output3, output4, output5, output6, output7, output8, output9, output10), 1)
        s, _ = torch.max(output, dim=-1)
        s, _ = torch.max(s, dim=-1)
        return s


def custom_loss_fn(model, criterion, y_pred, y, hp=1e-5):
    weights = [model.weight1, model.weight2, model.weight3, model.weight4, model.weight5, model.weight6, model.weight7,
               model.weight8, model.weight9, model.weight10]
    total_abs_diff = 0
    for c in range(len(weights)):
        block_size = 14
        weight = weights[c].reshape((196,196))
        v_size, h_size = weight.shape
        h_size = h_size // block_size
        v_size = v_size // block_size

        h_blocks = torch.hsplit(weight,h_size)
        v_blocks = torch.vstack(h_blocks)
        v_blocks = torch.vsplit(v_blocks,v_size*h_size)

        total_blocks = len(v_blocks)
        diff = 0
        current_block = v_blocks[0]
        for b in range(1,total_blocks):
            diff += torch.sum(torch.abs(v_blocks[b] - current_block))
            current_block = v_blocks[b]
        total_abs_diff += diff

    return criterion(y_pred, y) + hp * total_abs_diff


def train(model, criterion, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        prediction = model(data)
        loss = custom_loss_fn(model,criterion,prediction, target)
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

    model = Model2D().to(device)

    criterion = torch.nn.CrossEntropyLoss()  # loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer

    # training the model for 1000 epochs
    optimum_training_loss = float('inf')
    optimum_validation_loss = float('inf')

    # model.load_state_dict(torch.load("../models/2D_mnist_inter_block_sharing" + ".pt"))
    for epoch in range(40):
        training_loss = train(model,criterion,optimizer,train_loader,device)
        validation_loss = test(model,criterion,validation_loader,device,validation=True)
        print(f'training loss: {training_loss}, validation loss:{validation_loss}')
        if training_loss < optimum_training_loss and validation_loss < optimum_validation_loss:
            optimum_validation_loss = validation_loss
            optimum_training_loss = training_loss
            torch.save(model.state_dict(), "../models/2D_mnist_inter_block_sharing_14" + ".pt")

    model.load_state_dict(torch.load("../models/2D_mnist_inter_block_sharing_14" + ".pt"))
    accuracy = test(model,criterion,test_loader,device,validation=False)
    print(f'Test score: {sum(accuracy)/len(accuracy)}')


if __name__ == '__main__':
    main()