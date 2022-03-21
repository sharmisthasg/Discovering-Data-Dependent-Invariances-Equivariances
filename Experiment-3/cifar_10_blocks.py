# -*- coding: utf-8 -*-
"""CIFAR-10_Blocks.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1By_t1VmxWA3p-oUJBbUjjJVsRILYpV8Z
"""

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/content/drive')

# the correct path is important, while executing please put the path of the folder which will contain the hw3.dat file
# %cd /content/drive/MyDrive/696DS/

import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging
from torchsummary import summary
import itertools
import warnings
warnings.filterwarnings('always')
from collections import OrderedDict
import torchvision
import torch.utils.data as data
from torch.utils.data import DataLoader


def load_dataset():
    torch.manual_seed(696)
    train = torchvision.datasets.CIFAR10(
        root="./",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
                  torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    test = torchvision.datasets.CIFAR10(
        root="./",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
                  torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    # Random split with fixed seed
    train_set_size = int(len(train) * 0.8)
    valid_set_size = len(train) - train_set_size
    train, validation = data.random_split(train, [train_set_size, valid_set_size])

    print('Train data set:', len(train))
    print('Test data set:', len(test))
    print('Valid data set:', len(validation))

    batch_size = 256
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(dataset=validation, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader


def calculate_metric(metric_fn, true_y, pred_y):
    try:
        return metric_fn(true_y, pred_y, average="macro")
    except:
        return metric_fn(true_y, pred_y,)


def print_scores(p, r, f1, a, batch_size,logger):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        # print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")
        # logger.info(f"\t{name.rjust(14, ' ')}: {sum(scores) / batch_size:.4f}")
        print(f'{name}: {sum(scores)/len(scores)}')


def f(a):
    return torch.abs(a[0] - a[1])


def custom_loss_fn(criterion, y_pred, y, hp, model):
    blocks = torch.hsplit(model.linear_1.weight,192)
    h,w = blocks[0].shape
    v = torch.vstack(blocks)
    v = v.reshape(v.shape[0]*v.shape[1])
    m = v.reshape(v.shape[0] // (h*w), h*w)
    reg = 0
    for i in range(192):
        all_pairs = torch.combinations(m[i], r=2)
        combinations = torch.abs(all_pairs[:,0] - all_pairs[:,1])
        reg += torch.sum(combinations)

    return criterion(y_pred, y) + hp * reg


def train(model,criterion,optimizer,train_loader,device,hps):
    model.train()
    total_loss = 0
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        prediction = model(data)
        loss = custom_loss_fn(criterion, prediction, target, hps, model)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def validation(model,criterion,validation_loader,device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            prediction = model(data)
            loss = criterion(prediction, target)
            total_loss += loss

    return total_loss


def test(model,test_loader,device):
    model.eval()
    precision, recall, f1, accuracy = [], [], [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions = output.argmax(dim=1, keepdim=True)
            for acc, metric in zip((precision, recall, f1, accuracy),
                                   (precision_score, recall_score, f1_score, accuracy_score)):
                acc.append(
                    calculate_metric(metric, target.cpu(), predictions.cpu())
                )
            break
    return precision, recall, f1, accuracy


def main():
    torch.manual_seed(696)
    logging.basicConfig(filename="reports/2_layer_10_block_cifar_MLP.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    train_loader, validation_loader, test_loader = load_dataset()
    for images, labels in train_loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(OrderedDict([
        ('flatten', nn.Flatten()),
        ('linear_1', nn.Linear(3*32*32,5,bias=False)),
        ('relu', nn.ReLU()),
        ('linear_2', nn.Linear(5,10,bias=False))
    ])
    )

    print(summary(model=model,input_size=(3,32,32),batch_size=256))
    # print([sum(x.data) for x in model.linear.weight])

    model.to(device)
    lr = 1e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hps = [1e-6, 1e-5]
    optimum_loss = float('inf')
    optimum_val_loss = float('inf')

    # training-phase
    print('Training ....')
    for epoch in range(35):
        loss = train(model,criterion,optimizer,train_loader,device,hps[0])
        val_loss = validation(model, criterion, validation_loader, device)
        print(f'Epoch {epoch}: Training Loss {loss} Validation Loss {val_loss}')
        logger.info(f'Epoch {epoch}: Training Loss {loss} Validation Loss {val_loss}')
        if loss < optimum_loss and val_loss < optimum_val_loss :
            optimum_loss = loss
            optimum_val_loss = val_loss
            print('updating saved model')
            torch.save(model.state_dict(), "models/2_layer_10_block_cifar_MLP" + ".pt")

    # test-phase
    print('Testing ...')
    print('\nRESULTS ON TEST DATA:')
    logger.info('\nRESULTS ON TEST DATA:')
    model.load_state_dict(torch.load("models/2_layer_10_block_cifar_MLP" + ".pt"))
    precision, recall, f1, accuracy = test(model, test_loader, device)
    print_scores(precision, recall, f1, accuracy, len(test_loader),logger)


if __name__ == '__main__':
    main()

"""RESULTS ON TEST DATA (35 epochs):

precision: 0.26594436265488897
recall: 0.3130972070525823
F1: 0.2842120908633986
accuracy: 0.32421875
"""