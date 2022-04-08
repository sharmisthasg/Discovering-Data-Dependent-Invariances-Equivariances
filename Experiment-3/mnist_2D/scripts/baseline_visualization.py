import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
from numpy.linalg import matrix_rank
from collections import OrderedDict
import seaborn as sns
import numpy as np
import torch.nn.functional as F


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


def main():
    model = Baseline()
    model.load_state_dict(torch.load("../models/2D_mnist_baseline_aug" + ".pt"))

    weights = model.conv1.weight
    for c in range(10):
        weight = weights[c]
        weight = torch.reshape(weight, (weight.shape[-1], weight.shape[-2]))
        sns.heatmap(weight.detach().numpy())
        plt.show()




if __name__ == '__main__':
    main()