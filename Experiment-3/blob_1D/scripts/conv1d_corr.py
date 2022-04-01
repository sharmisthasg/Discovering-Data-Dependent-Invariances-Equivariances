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

class Model1D(torch.nn.Module):
    def __init__(self):
        super(Model1D, self).__init__()
        weight = torch.randn(5,10,10)
        self.weight1 = torch.nn.Parameter(weight)
        torch.nn.init.kaiming_normal_(self.weight1)
        torch.nn.init.kaiming_normal_(self.weight1)
        self.bias1 = torch.nn.Parameter(torch.randn(10))

    def forward(self,x):
        layer1_op = torch.matmul(x,self.weight1)  # (N,10) x (10,10,5) = (5, N, 10)
        layer1_op = torch.tanh(layer1_op)
        layer1_op = torch.transpose(layer1_op,0,1)  # (5, N, 10) --> (N, 5, 10)
        dim0, dim1, dim2 = layer1_op.shape
        prediction = torch.sum(layer1_op, dim=2).reshape(dim0,dim1)  # (N, 50) x (50, 5) --> (N, 5)
        return prediction


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
    model = Model1D()
    baseline = CNN_1D()
    model.load_state_dict(torch.load("../models/1D_model_sparsity_bias" + ".pt"))
    baseline.load_state_dict(torch.load("../models/1D_baseline" + ".pt"))
    weight = model.weight1
    bweight = baseline.conv1d.weight

    corr_scores = [-1] * 5
    for c in range(5):
        corr = -1
        bw = bweight[c].detach().numpy()
        mw = weight[c].detach().numpy()
        for i in range(10):
            for j in range(7):
                corr = max(corr, np.corrcoef(bw,mw[i][j:j+4])[0][1])
        corr_scores[c] = corr

    print(corr_scores)




if __name__ == '__main__':
    main()