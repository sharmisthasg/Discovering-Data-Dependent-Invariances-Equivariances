import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
from numpy.linalg import matrix_rank
from collections import OrderedDict
import seaborn as sns
import numpy as np
import torch.nn.functional as F
#from torchsummary import summary

class Conv2DMaxpool(torch.nn.Module):
    def __init__(self):
        super(Conv2DMaxpool, self).__init__()
        weight1 = torch.randn(1, 1, 100, 100)
        self.weight1 = torch.nn.Parameter(weight1)
        torch.nn.init.kaiming_normal_(self.weight1)
        weight2 = torch.randn(1, 1, 100, 100)
        self.weight2 = torch.nn.Parameter(weight2)
        torch.nn.init.kaiming_normal_(self.weight2)
        weight3 = torch.randn(1, 1, 100, 100)
        self.weight3 = torch.nn.Parameter(weight3)
        torch.nn.init.kaiming_normal_(self.weight3)

    def forward(self, x):
        x = torch.reshape(x, (1, 1, x.shape[0], x.shape[1]))
        output1 = F.conv2d(self.weight1, x, stride=(10, 10))
        output2 = F.conv2d(self.weight2, x, stride=(10, 10))
        output3 = F.conv2d(self.weight3, x, stride=(10, 10))
        output = torch.cat((output1, output2, output3), 1)
        s, _ = torch.max(output, dim=-1)
        s, _ = torch.max(s, dim=-1)
        prediction = F.softmax(s, dim=1)
        return prediction

class Conv2DCnn(torch.nn.Module):
    def __init__(self):
        super(Conv2DCnn, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 5)

    def forward(self, x):
        x = x.reshape(x.size(dim=0), 1, x.size(dim=1), x.size(dim=2))
        x = self.conv1(x)
        x = torch.tanh(x)
        x,_ = torch.max(x, dim=-1)
        x,_ = torch.max(x, dim=-1)
        x = torch.nn.functional.softmax(x, dim=1)
        return x

def main():
    torch.manual_seed(696)
    model = Conv2DCnn()
    current_model = "../models/conv2d_cnn"
    model.load_state_dict(torch.load(current_model + '.pt', map_location='cpu'))

    weight1 = model.conv1.weight[0, :, :]
    weight2 = model.conv1.weight[1, :, :]
    weight3 = model.conv1.weight[2, :, :]
    weight1 = torch.reshape(weight1, (weight1.shape[-1], weight1.shape[-2]))
    weight2 = torch.reshape(weight2, (weight2.shape[-1], weight2.shape[-2]))
    weight3 = torch.reshape(weight3, (weight3.shape[-1], weight3.shape[-2]))
    weight1 = weight1.detach().numpy()
    weight2 = weight2.detach().numpy()
    weight3 = weight3.detach().numpy()

    model = Conv2DMaxpool()
    current_model = "../models/conv2d_maxpool"
    model.load_state_dict(torch.load(current_model + '.pt', map_location='cpu'))

    maxpool_weight1 = model.weight1
    maxpool_weight2 = model.weight2
    maxpool_weight3 = model.weight3

    maxpool_weight1 = torch.squeeze(maxpool_weight1).detach().numpy()
    maxpool_weight2 = torch.squeeze(maxpool_weight2).detach().numpy()
    maxpool_weight3 = torch.squeeze(maxpool_weight3).detach().numpy()

    corr1 = -1
    for i in range(0, 100, 10):
        for j in range(0, 100, 10):
            for k in range(i, i+6):
                for l in range(j, j+6):
                    patch = maxpool_weight1[k:k+5, l:l+5].reshape(1, -1)
                    #print(patch.shape, maxpool_weight1.shape)
                    corr1 = max(corr1, np.corrcoef(patch, weight1.reshape(1, -1))[0,1])

    corr2 = -1
    for i in range(0, 100, 10):
        for j in range(0, 100, 10):
            for k in range(i, i+6):
                for l in range(j, j+6):
                    patch = maxpool_weight2[k:k+5, l:l+5].reshape(1, -1)
                    corr2 = max(corr2, np.corrcoef(patch, weight2.reshape(1, -1))[0,1])

    corr3 = -1
    for i in range(0, 100, 10):
        for j in range(0, 100, 10):
            for k in range(i, i+6):
                for l in range(j, j+6):
                    patch = maxpool_weight3[k:k+5, l:l+5].reshape(1, -1)
                    corr3 = max(corr3, np.corrcoef(patch, weight3.reshape(1, -1))[0,1])

    print(f"Higest correlations for the 3 weight layers with the cnn kernels are {corr1}, {corr2}, {corr3} for class line, triangle, and square respectively")

if __name__ == '__main__':
    main()
