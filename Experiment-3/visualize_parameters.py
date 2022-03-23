import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
from numpy.linalg import matrix_rank
from collections import OrderedDict
import numpy as np

def main():
    # Define the model arch or instantiate an object of the model's class and write the class in the script
    model = nn.Sequential(OrderedDict([
        ('flatten', nn.Flatten()),
        ('linear_1', nn.Linear(28 * 28, 128, bias=False)),
        ('relu', nn.ReLU()),
        ('linear_2', nn.Linear(128, 10, bias=False))
    ])
    )

    # give the correct path of the model you want to visualize
    current_model = "./models/H_V_block_MNIST"
    model.load_state_dict(torch.load(current_model + '.pt'))

    layer1 = model.linear_1.weight.detach().numpy().reshape((128,784))
    # plt.imshow(layer1,aspect='auto')
    # plt.show()

    #zoom in
    zoomparam = 128
    zoomed = np.zeros((zoomparam,zoomparam))
    for i in range(zoomparam):
        for j in range(zoomparam):
            zoomed[i][j] = layer1[i][j]
    # print(zoomed)
    # plt.imshow(zoomed, aspect='auto')
    # plt.show()

    #correlation
    coeffdict = {}
    for i in range(128):
        for j in range(28):
            if i!=j:
                corrmatrix = np.corrcoef(layer1[i], layer1[j])
                coeffdict[str(i)+", "+str(j)] = corrmatrix[0][1]
    # print(coeffdict)
    values = list(coeffdict.values())
    maxcorr = max(values)
    print("Maximum Correlation", maxcorr)
    items = coeffdict.items()
    for key, value in items:
        if value == maxcorr:
            print("Corresponding Rows: ", key)

    #enter rows to compare
    rows = input("Enter indices of rows separated by comma").split(",")
    for row in rows:
        plt.plot(range(784), layer1[int(row)], label="row"+row)
    plt.legend()
    plt.savefig('row_comparisons/H_V_block_MNIST_'+str(rows)+'.png')
    plt.show()


    # first_row = torch.clone(model.linear_1.weight[0])
    # second_row = torch.clone(model.linear_1.weight[1])
    # # print(torch.linalg.norm(first_row))
    # # print(first_row)
    # first_row = torch.div(first_row,torch.norm(first_row).expand_as(first_row))
    # second_row = torch.div(second_row, torch.norm(second_row).expand_as(second_row))
    # # print(first_row.dot(second_row))


if __name__ == '__main__':
    main()
