import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
from numpy.linalg import matrix_rank
from collections import OrderedDict
import seaborn as sns
import numpy as np
import torch.nn.functional as F


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



def main():
    model = Model2D()
    model.load_state_dict(torch.load("../models/2D_mnist_conv2d_tanh_aug" + ".pt"))

    weights = [model.weight1, model.weight2, model.weight3, model.weight4, model.weight5, model.weight6, model.weight7,
               model.weight8, model.weight9, model.weight10]
    for c in range(10):
        weight = weights[c]
        weight = torch.reshape(weight, (weight.shape[-1], weight.shape[-2]))
        sns.heatmap(weight.detach().numpy())
        plt.show()



if __name__ == '__main__':
    main()