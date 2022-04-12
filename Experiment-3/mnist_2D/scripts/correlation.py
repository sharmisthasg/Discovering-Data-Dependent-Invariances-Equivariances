import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
from numpy.linalg import matrix_rank
from collections import OrderedDict
import seaborn as sns
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging
import warnings
warnings.filterwarnings('always')
import sys
sys.path.append('../datasets')
import Load_augmented_dataset


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


class CNN_Modified(torch.nn.Module):
    def __init__(self, weight):
        super(CNN_Modified, self).__init__()
        self.patch_kernel = torch.nn.Parameter(weight.reshape((10,1,28,28)))

    def forward(self, x):
        x = x.double()
        x = F.conv2d(x,self.patch_kernel)
        x = torch.tanh(x)
        x, _ = torch.max(x, dim=-1)
        x, _ = torch.max(x, dim=-1)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


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
    model = Model2D()
    model.load_state_dict(torch.load("../models/2D_mnist_inter_block_sharing" + ".pt"))
    cnn = Baseline()
    cnn.load_state_dict(torch.load("../models/2D_mnist_baseline" + ".pt"))

    weights = [model.weight1, model.weight2, model.weight3, model.weight4, model.weight5, model.weight6, model.weight7,
               model.weight8, model.weight9, model.weight10]
    cnn_weight = cnn.conv1.weight.detach().numpy()
    corr = [0] * 10
    best_patch = [0] * 10
    for c in range(10):
        weight = weights[c]
        weight = torch.reshape(weight, (weight.shape[-1], weight.shape[-2])).detach().numpy()
        for i in range(0,169):
            for j in range(0,169):
                patch = weight[i:i+28,j:j+28]
                current_corr = np.corrcoef(patch, cnn_weight[c].reshape(28, 28))[0,1]
                if current_corr > corr[c]:
                    corr[c] = current_corr
                    best_patch[c] = patch

    print('max correlation scores:\n', corr)

    patch_kernel = np.empty((10,28,28))
    for c in range(len(best_patch)):
        patch_kernel[c] = best_patch[c]

    # checking accuracy on test data using MLP patch in CNN
    train_loader, validation_loader, test_loader = Load_augmented_dataset.main()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn_modified = CNN_Modified(torch.from_numpy(patch_kernel)).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    accuracy = test(cnn_modified, criterion, test_loader, device, validation=False)
    print(f'Test score: {sum(accuracy) / len(accuracy)}')

    # plotting the best patches
    for c in range(10):
        weight = patch_kernel[c]
        sns.heatmap(weight)
        plt.show()


if __name__ == '__main__':
    main()