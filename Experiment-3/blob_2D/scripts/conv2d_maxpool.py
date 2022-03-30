import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging
import numpy as np
import warnings
warnings.filterwarnings('always')
import dataset_2D_generator
from collections import OrderedDict
from torch.nn import functional as F

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
        #print(s)
        prediction = F.softmax(s, dim=1)
        #print(prediction)
        return prediction

def load_dataset():
    samples, targets = dataset_2D_generator.main()
    shuffled_indices = np.random.permutation(samples.shape[0])
    shuffled_samples = samples[shuffled_indices]
    shuffled_targets = targets[shuffled_indices]

    train_size = int(0.7 * len(shuffled_samples))
    validation_size = train_size + int(0.2 * len(shuffled_samples))

    X_train, y_train = shuffled_samples[:train_size], shuffled_targets[:train_size]
    X_validation, y_validation = shuffled_samples[train_size:validation_size], shuffled_targets[train_size:validation_size]
    X_test, y_test = shuffled_samples[validation_size:], shuffled_targets[validation_size:]

    return {"train": dict(X=torch.Tensor(X_train), y=torch.Tensor(y_train)),
            "test": dict(X=torch.Tensor(X_test), y=torch.Tensor(y_test)),
            "validation": dict(X=torch.Tensor(X_validation), y=torch.Tensor(y_validation))}


def train(model,criterion,optimizer,X,y):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()  # backprop (compute gradients)
    optimizer.step()  # update weights (gradient descent step)
    optimizer.zero_grad()  # reset gradients
    return loss


def test(model,criterion,X,y,validation):
    with torch.no_grad():
        y_pred = model(X)  # forward step
        if validation:
            eval_metric = criterion(y_pred, y)  # compute loss
            return eval_metric
        else:
            #print("y_pred", y_pred)
            return torch.argmax(y_pred,dim=1)

def main():
    torch.manual_seed(696)
    data = load_dataset()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train, y_train = data['train']['X'].to(device), data['train']['y'].type(torch.LongTensor).to(device)
    X_validation, y_validation = data['validation']['X'].to(device), data['validation']['y'].type(torch.LongTensor).to(device)
    X_test, y_test = data['test']['X'].to(device), data['test']['y'].type(torch.LongTensor).to(device)
    lr = 5e-2  # Learning rate

    model = Model2D().to(device)

    criterion = torch.nn.CrossEntropyLoss()  # loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer

    # training the model for 1000 epochs
    optimum_training_loss = float('inf')
    optimum_validation_loss = float('inf')

    for epoch in range(5):
        training_loss = 0
        validation_loss = 0
        model.train()
        for i in range(X_train.shape[0]):
            training_loss += train(model,criterion,optimizer,X_train[i],y_train[i].reshape((1))).data.cpu()
        model.eval()
        for i in range(X_validation.shape[0]):
            validation_loss += test(model,criterion,X_validation[i],y_validation[i].reshape((1)),validation=True).detach().data.cpu()
        training_loss = training_loss/X_train.shape[0]
        validation_loss = validation_loss/X_validation.shape[0]
        print(f'training loss: {training_loss}, validation loss:{validation_loss}')
        if training_loss < optimum_training_loss and validation_loss < optimum_validation_loss:
            optimum_validation_loss = validation_loss
            optimum_training_loss = training_loss
            torch.save(model.state_dict(), "./models/conv2d_maxpool.pt")

    model.load_state_dict(torch.load("./models/conv2d_maxpool.pt"))
    accuracy = 0
    model.eval()
    for i in range(X_test.shape[0]):
        pred = test(model, criterion, X_test[i], y_test[i], validation=False)
        #print(pred, y_test[i])
        if pred == y_test[i]:
            accuracy += 1
    accuracy = accuracy/X_test.shape[0]
    print(f'Test accuracy: {accuracy}')

if __name__ == '__main__':
    main()
