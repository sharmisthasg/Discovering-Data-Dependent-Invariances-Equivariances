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

    def forward(self,x):
        layer1_op = torch.matmul(x,self.weight1)  # (N,10) x (10,10,5) = (5, N, 10)
        layer1_op = torch.sigmoid(layer1_op)
        layer1_op = torch.transpose(layer1_op,0,1)  # (5, N, 10) --> (N, 5, 10)
        dim0, dim1, dim2 = layer1_op.shape
        prediction = torch.sum(layer1_op, dim=2).reshape(dim0,dim1)  # (N, 50) x (50, 5) --> (N, 5)
        return prediction

'''class Model1D(torch.nn.Module):
    def __init__(self):
        super(Model1D, self).__init__()
        self.weight1 = torch.nn.Parameter(torch.randn(5,10,10))
        torch.nn.init.kaiming_normal_(self.weight1)
        self.weight2 = torch.nn.Parameter(torch.randn(50, 5))
        torch.nn.init.kaiming_normal_(self.weight2)

    def forward(self,x):
        layer1_op = torch.matmul(x,self.weight1)  # (N,10) x (10,10,5) = (5, N, 10)
        layer1_op = F.relu(layer1_op)
        layer1_op = torch.transpose(layer1_op,0,1)  # (5, N, 10) --> (N, 5, 10)
        dim0, dim1, dim2 = layer1_op.shape
        layer2_op = layer1_op.reshape((dim0, dim1 * dim2))
        prediction = torch.matmul(layer2_op, self.weight2)
        return prediction'''


def main():
    torch.manual_seed(696)
    model = Model1D()
    current_model = "./models/1D_model"
    model.load_state_dict(torch.load(current_model + '.pt', map_location='cpu'))

    weight1 = model.weight1.T
    #weight2 = model.weight2.T

    for category in range(5):
        '''ax = sns.heatmap(weight1[category].detach().numpy())
        plt.title('class: ' + str(category))
        plt.savefig("Weight Heatmaps/1D_model_class_" + str(category) + ".png")
        plt.clf()'''
    max_correlation = {}
    for i in range(5):
        coeffdict = {}
        for row1 in range(10):
            k = [m for m in range(10)]
            #k.remove(row1)
            fig, axs = plt.subplots(ncols=1, nrows=10, figsize = (5, 40), constrained_layout=True)
            for row2 in k:
                #coeffdict[str(row1) + ", " + str(row2)] = -float('inf')
                shifts = []
                correlations = []
                for j in range(10):
                    temp = weight1[i][row2].detach().numpy()
                    temp = np.hstack((temp[-j:], temp[:-j]))
                    corrmatrix = np.corrcoef(weight1[i][row1].detach().numpy(), temp)
                    correlation = corrmatrix[0][1]
                    shifts.append(j)
                    correlations.append(correlation)
                    #coeffdict[str(row1) + ", " + str(row2)] = max(coeffdict[str(row1) + ", " + str(row2)], corrmatrix[0][1])
                axs[row2].scatter(shifts, correlations)
                axs[row2].set_title("Row {} vs Row {} correlations vs shifts".format(row1, row2))
                print("Best correltion for row {} and row {} is {} for shift {}".format(row1, row2, max(correlations), correlations.index(max(correlations))))
            fig.suptitle("Shift vs Correlation for row {}".format(row1))
            plt.savefig("Weight Heatmaps/1D_model_corr_class_" + str(i) + "_row_" + str(row1)+".png")
            plt.clf()
        print(f'Class: {i}')
        '''values = list(coeffdict.values())
        maxcorr = max(values)
        print("Maximum Correlation", maxcorr)
        items = coeffdict.items()
        for key, value in items:
            if value == maxcorr:
                print("Corresponding Rows: ", key)
                max_correlation[i] = list()
                r1, r2 = str(key).split(',')
                max_correlation[i] = [int(r1), int(r2)]
                max_correlation[i].append(value)'''

    '''for category in range(5):
        row1 = weight1[category][max_correlation[category][0]].detach().numpy()
        row2 = weight1[category][max_correlation[category][1]].detach().numpy()
        plt.plot(range(10), row1, label='row-'+str(max_correlation[category][0]))
        plt.plot(range(10), row2, label='row-'+str(max_correlation[category][1]))
        plt.legend()
        plt.title('Correlation coeff for class ' + str(category) + ': ' + str(max_correlation[category][2]))
        plt.savefig("Weight Heatmaps/1D_model_coeff_class_" + str(category) + ".png")
        plt.clf()'''

if __name__ == '__main__':
    main()
