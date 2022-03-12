import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
from numpy.linalg import matrix_rank


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
    D = 10
    C = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define the model arch or instantiate an object of the model's class and write the class in the script
    '''model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(100 * 100, 64, bias=False),
        nn.ReLU(),
        nn.Linear(64, 32, bias=False),
        nn.ReLU(),
        nn.Linear(32, 10)
    )'''
    model = torch.nn.Sequential(torch.nn.Linear(D, D, bias=False), torch.nn.ReLU(), torch.nn.Linear(D, C, bias=False), torch.nn.ReLU())
    model.to(device)
    # give the correct path of the model you want to visualize
    current_model = "models/one_hidden_layer/best_validation"
    model.load_state_dict(torch.load(current_model + '.pt'))

    for name, parameter in model.named_parameters():
        if 'bias' not in name:
            print(name)
            try:
                plt.imshow(parameter[0].cpu().detach().numpy().transpose(1,2,0), aspect='auto')
            except:
                plt.imshow(parameter.cpu().detach().numpy(), aspect='auto')
                print(matrix_rank(parameter.cpu().detach().numpy()))
        else:
            plt.imshow(parameter.cpu().detach().numpy().reshape((parameter.shape[0],1)), aspect='auto')
            print(matrix_rank(parameter.cpu().detach().numpy()))
        plt.show()
        plt.savefig("heatmaps/one_hidden_layer/best_validation_" + str(name) + "_reg.png")

if __name__ == '__main__':
    main()
