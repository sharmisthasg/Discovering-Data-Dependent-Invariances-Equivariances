import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
from numpy.linalg import matrix_rank
from collections import OrderedDict
import seaborn as sns
import numpy as np
import torch.nn.functional as F
from torchsummary import summary


class Model1D(torch.nn.Module):
    def __init__(self):
        super(Model1D, self).__init__()
        self.weight1 = torch.nn.Parameter(torch.randn(5,10,10))
        torch.nn.init.kaiming_normal_(self.weight1)
        self.weight2 = torch.nn.Parameter(torch.randn(10, 1))
        torch.nn.init.kaiming_normal_(self.weight2)

    def forward(self,x):
        layer1_op = torch.matmul(x,self.weight1)  # (N,10) x (10,10,5) = (5, N, 10)
        layer1_op = F.relu(layer1_op)
        layer1_op = torch.transpose(layer1_op,0,1)  # (5, N, 10) --> (N, 5, 10)
        dim0, dim1, dim2 = layer1_op.shape
        prediction = torch.matmul(layer1_op, self.weight2).reshape((dim0, dim1))
        return prediction


def main():
    torch.manual_seed(696)
    model = Model1D()
    current_model = "./models/1D_2_model"
    model.load_state_dict(torch.load(current_model + '.pt', map_location='cpu'))

    weight1 = model.weight1
    weight2 = model.weight2

    for category in range(5):
        ax = sns.heatmap(weight1[category].detach().numpy().transpose())
        plt.title('class: ' + str(category))
        plt.show()

    max_correlation = {}
    for i in range(5):
        coeffdict = {}
        for row1 in range(10):
            k = [m for m in range(10)]
            k.remove(row1)
            for row2 in k:
                corrmatrix = np.corrcoef(weight1[i][row1].detach().numpy(), weight1[i][row2].detach().numpy())
                coeffdict[str(row1) + ", " + str(row2)] = corrmatrix[0][1]
        print(f'Class: {i}')
        values = list(coeffdict.values())
        maxcorr = max(values)
        print("Maximum Correlation", maxcorr)
        items = coeffdict.items()
        for key, value in items:
            if value == maxcorr:
                print("Corresponding Rows: ", key)
                max_correlation[i] = list()
                r1, r2 = str(key).split(',')
                max_correlation[i] = [int(r1), int(r2)]
                max_correlation[i].append(value)

    for category in range(5):
        row1 = weight1[category][max_correlation[category][0]].detach().numpy()
        row2 = weight1[category][max_correlation[category][1]].detach().numpy()
        plt.plot(range(10), row1, label='row-'+str(max_correlation[category][0]))
        plt.plot(range(10), row2, label='row-'+str(max_correlation[category][1]))
        plt.legend()
        plt.title('Correlation coeff for class ' + str(category) + ': ' + str(max_correlation[category][2]))
        plt.show()


if __name__ == '__main__':
    main()