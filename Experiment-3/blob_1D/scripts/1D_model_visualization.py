import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
from numpy.linalg import matrix_rank
from collections import OrderedDict
import numpy as np
import seaborn as sns


class Model1D(torch.nn.Module):
    def __init__(self):
        super(Model1D, self).__init__()
        weight = torch.randn(5,10,10)
        self.weight1 = torch.nn.Parameter(weight)
        # torch.nn.init.kaiming_normal_(self.weight1)
        # self.bias1 = torch.nn.Parameter(torch.randn(10))

    def forward(self,x):
        layer1_op = torch.matmul(x,self.weight1) + self.bias1
        layer1_op = torch.sigmoid(layer1_op)
        layer1_op = torch.transpose(layer1_op,0,1)
        dim0, dim1, dim2 = layer1_op.shape
        prediction = torch.sum(layer1_op, dim=2).reshape(dim0,dim1)
        return prediction


def main():
    model = Model1D()
    model.load_state_dict(torch.load("../models/1D_model_tanh" + ".pt"))
    weight = model.weight1.transpose(1,2)
    for channel in range(weight.shape[0]):
        sns.heatmap(weight[channel].detach().numpy())
        plt.show()


if __name__ == '__main__':
    main()
