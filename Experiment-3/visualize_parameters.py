import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
from numpy.linalg import matrix_rank
from collections import OrderedDict


class baseline(nn.Module):
    def __init__(self):
        super(baseline, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=4, stride=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=4, stride=2, bias=False),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(1 * 23 * 23, 10),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # print("conv output", x.shape)
        x = x.view(-1, 1 * 23 * 23)
        # print("fc input", x.shape)
        x = self.fc_layers(x)
        return x


def main():
    # Define the model arch or instantiate an object of the model's class and write the class in the script
    model = nn.Sequential(OrderedDict([
        ('flatten', nn.Flatten()),
        ('linear', nn.Linear(100 * 100, 2, bias=False)),
    ])
    )

    # give the correct path of the model you want to visualize
    current_model = "./models/1_layer_0-1_MLP"
    model.load_state_dict(torch.load(current_model + '.pt'))

    # for name, parameter in model.named_parameters():
    #     if 'bias' not in name:
    #         print(name)
    #         try:
    #             plt.imshow(parameter[0].detach().numpy().transpose(1,2,0), aspect='auto')
    #         except:
    #             plt.imshow(parameter[0].detach().numpy(), aspect='auto')
    #             print(matrix_rank(parameter.detach().numpy()))
    #     else:
    #         plt.imshow(parameter.detach().numpy().reshape((parameter.shape[0],1)), aspect='auto')
    #         print(matrix_rank(parameter.detach().numpy()))
    #     plt.show()

    plt.imshow(model.linear.weight[0].detach().numpy().reshape((100,100)),aspect='auto')
    plt.show()
    first_row = torch.clone(model.linear.weight[0])
    second_row = torch.clone(model.linear.weight[1])
    # print(torch.linalg.norm(first_row))
    # print(first_row)
    first_row = torch.div(first_row,torch.norm(first_row).expand_as(first_row))
    second_row = torch.div(second_row, torch.norm(second_row).expand_as(second_row))
    print(first_row.dot(second_row))


if __name__ == '__main__':
    main()
