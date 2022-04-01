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
    model = Model2D()
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

    image = plt.imshow(weight1)
    plt.colorbar(image)
    plt.savefig('../visualizations/conv2d_cnn/weight1.png')
    plt.clf()
    
    image = plt.imshow(weight2)
    plt.colorbar(image)
    plt.savefig('../visualizations/conv2d_cnn/weight2.png')
    plt.clf()

    image = plt.imshow(weight3)
    plt.colorbar(image)
    plt.savefig('../visualizations/conv2d_cnn/weight3.png')
    plt.clf()

if __name__ == '__main__':
    main()
