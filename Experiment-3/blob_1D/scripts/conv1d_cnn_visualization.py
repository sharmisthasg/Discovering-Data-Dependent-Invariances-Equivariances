import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
from numpy.linalg import matrix_rank
from collections import OrderedDict
import numpy as np
import seaborn as sns


class CNN_1D(torch.nn.Module):
    def __init__(self):
        super(CNN_1D, self).__init__()
        self.conv1d = torch.nn.Conv1d(1,5,4)

    def forward(self,x):
        dim0, dim1 = x.shape
        x = x.reshape(dim0,1,dim1)
        x = self.conv1d(x)
        x = torch.tanh(x)
        x, x_ind = torch.max(x,dim=-1)
        return x

def main():
    model = CNN_1D()
    model.load_state_dict(torch.load("../models/1D_baseline" + ".pt"))
    weight = model.conv1d.weight
    for channel in range(weight.shape[0]):
        sns.heatmap(weight[channel].detach().numpy())
        plt.show()


if __name__ == '__main__':
    main()
