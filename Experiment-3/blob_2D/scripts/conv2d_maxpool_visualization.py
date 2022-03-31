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

class Model2D(torch.nn.Module):
    def __init__(self):
        super(Model2D, self).__init__()
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

def main():
    torch.manual_seed(696)
    model = Model2D()
    current_model = "../models/conv2d_maxpool"
    model.load_state_dict(torch.load(current_model + '.pt', map_location='cpu'))

    weight1 = model.weight1
    weight2 = model.weight2
    weight3 = model.weight3
    weight1 = torch.reshape(weight1, (weight1.shape[-1], weight1.shape[-2]))
    weight2 = torch.reshape(weight2, (weight2.shape[-1], weight2.shape[-2]))
    weight3 = torch.reshape(weight3, (weight3.shape[-1], weight3.shape[-2]))
    weight1 = weight1.detach().numpy()
    weight2 = weight2.detach().numpy()
    weight3 = weight3.detach().numpy()

    image = plt.imshow(weight1)
    plt.colorbar(image)
    plt.savefig('../visualizations/conv2d_maxpool/weight1.png')
    plt.clf()
    
    image = plt.imshow(weight2)
    plt.colorbar(image)
    plt.savefig('../visualizations/conv2d_maxpool/weight2.png')
    plt.clf()

    image = plt.imshow(weight3)
    plt.colorbar(image)
    plt.savefig('../visualizations/conv2d_maxpool/weight3.png')
    plt.clf()
if __name__ == '__main__':
    main()
