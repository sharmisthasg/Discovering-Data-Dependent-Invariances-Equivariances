import torch
import torchvision
import torch.utils.data as data
from torch.utils.data import DataLoader


class Sampler(torch.utils.data.sampler.Sampler):
    def __init__(self, mask, data_source):
        self.mask = mask
        self.data_source = data_source

    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(self.mask)])

    def __len__(self):
        return len(self.data_source)


def main():
    torch.manual_seed(696)
    train = torchvision.datasets.MNIST(
        root="./",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
                  torchvision.transforms.Resize((100,100)),
                  torchvision.transforms.ToTensor()
        ]))
    test = torchvision.datasets.MNIST(
        root="./",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
                  torchvision.transforms.Resize((100,100)),
                  torchvision.transforms.ToTensor()
        ]))

    # Random split with fixed seed
    train_set_size = int(len(train) * 0.8)
    valid_set_size = len(train) - train_set_size
    train, validation = data.random_split(train, [train_set_size, valid_set_size])

    mask_train = torch.tensor([1 if train[i][1] == 6 or train[i][1] == 9 else 0 for i in range(len(train))])
    train_sampler = Sampler(mask_train, train)
    # train.targets[train.targets == 6] = 0
    # train.targets[train.targets == 9] = 1

    mask_validation = torch.tensor([1 if validation[i][1] == 6 or validation[i][1] == 9 else 0 for i in range(len(validation))])
    validation_sampler = Sampler(mask_validation, validation)
    # validation.targets[validation.targets == 6] = 0
    # validation.targets[validation.targets == 9] = 1

    mask_test = torch.tensor([1 if test[i][1] == 6 or test[i][1] == 9 else 0 for i in range(len(test))])
    test_sampler = Sampler(mask_test, test)
    # test.targets[test.targets == 6] = 0
    # test.targets[test.targets == 9] = 1

    batch_size = 128
    train_loader = DataLoader(dataset=train, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, sampler=test_sampler)
    validation_loader = DataLoader(dataset=validation, batch_size=batch_size, sampler=validation_sampler)

    return train_loader, validation_loader, test_loader


if __name__ == '__main__':
    main()


