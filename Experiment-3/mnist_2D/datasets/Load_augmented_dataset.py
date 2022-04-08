import torch
import torchvision
import torch.utils.data as data
from torch.utils.data import DataLoader


def main():
    torch.manual_seed(696)
    train = torchvision.datasets.MNIST(
        root="./",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
                  torchvision.transforms.ToTensor(),
                  torchvision.transforms.RandomAffine(degrees=0, translate=(0.2, 0))
        ]))
    test = torchvision.datasets.MNIST(
        root="./",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
                  torchvision.transforms.ToTensor(),
                  torchvision.transforms.RandomAffine(degrees=0, translate=(0.2, 0))
        ]))

    # Random split with fixed seed
    train_set_size = int(len(train) * 0.8)
    valid_set_size = len(train) - train_set_size
    train, validation = data.random_split(train, [train_set_size, valid_set_size])

    print('Train data set:', len(train))
    print('Test data set:', len(test))
    print('Valid data set:', len(validation))

    batch_size = 256
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(dataset=validation, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader


if __name__ == '__main__':
    main()


