import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import warnings
from Dataset import Load_dataset
warnings.filterwarnings('always')


class Model2D(torch.nn.Module):
    def __init__(self):
        super(Model2D, self).__init__()
        weight = torch.randn(10,784,784)
        self.weight1 = torch.nn.Parameter(weight)
        torch.nn.init.kaiming_normal_(self.weight1)

    def forward(self,x):
        x_dim0,x_dim1, x_dim2, x_dim3 = x.shape
        x = x.reshape(x_dim0, x_dim1 * x_dim2 * x_dim3)
        layer1_op = torch.matmul(x,self.weight1)
        layer1_op = torch.sigmoid(layer1_op)
        layer1_op = torch.transpose(layer1_op,0,1)
        dim0, dim1, dim2 = layer1_op.shape
        prediction = torch.sum(layer1_op, dim=2).reshape(dim0,dim1)
        return prediction


def custom_loss(y_pred, y, criterion, model):
    hp = 1e-5
    return criterion(y_pred, y) + hp * torch.sum(torch.abs(model.weight1))


def train(model, criterion, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        prediction = model(data)
        loss = custom_loss(prediction, target, criterion, model)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def test(model,criterion,loader,device,validation):
    model.eval()
    total_loss = 0
    accuracy = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            prediction = model(data)
            if validation:
                loss = criterion(prediction, target)
                total_loss += loss
            else:
                predictions = prediction.argmax(dim=1, keepdim=True)
                accuracy.append(accuracy_score(target.cpu(), predictions.cpu()))
    if validation:
        return total_loss
    else:
        return accuracy


def main():
    torch.manual_seed(696)
    # data = load_dataset()
    #
    # X_train, y_train = data['train']['X'], data['train']['y'].type(torch.LongTensor)
    # X_validation, y_validation = data['validation']['X'], data['validation']['y'].type(torch.LongTensor)
    # X_test, y_test = data['test']['X'], data['test']['y'].type(torch.LongTensor)

    train_loader, validation_loader, test_loader = Load_dataset.main()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lr = 5e-3  # Learning rate

    model = Model2D().to(device)

    criterion = torch.nn.CrossEntropyLoss()  # loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer

    # training the model for 1000 epochs
    optimum_training_loss = float('inf')
    optimum_validation_loss = float('inf')

    for epoch in range(50):
        training_loss = train(model,criterion,optimizer,train_loader,device)
        validation_loss = test(model,criterion,validation_loader,device,validation=True)
        print(f'training loss: {training_loss}, validation loss:{validation_loss}')
        if training_loss < optimum_training_loss and validation_loss < optimum_validation_loss:
            optimum_validation_loss = validation_loss
            optimum_training_loss = training_loss
            torch.save(model.state_dict(), "./models/2D_mnist" + ".pt")

    model.load_state_dict(torch.load("./models/2D_mnist" + ".pt"))
    accuracy = test(model,criterion,test_loader,device,validation=False)
    print(f'Test score: {sum(accuracy)/len(accuracy)}')


if __name__ == '__main__':
    main()