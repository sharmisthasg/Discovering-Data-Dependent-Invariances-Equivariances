import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import warnings
warnings.filterwarnings('always')


class Model2D(torch.nn.Module):
    def __init__(self):
        super(Model2D, self).__init__()
        weight = torch.randn(3,100,100)
        self.weight1 = torch.nn.Parameter(weight)
        torch.nn.init.kaiming_normal_(self.weight1)

    def forward(self,x):
        x_dim0,x_dim1, x_dim2 = x.shape
        x = x.reshape(x_dim0, x_dim1 * x_dim2)
        layer1_op = torch.matmul(x,self.weight1)
        layer1_op = torch.sigmoid(layer1_op)
        layer1_op = torch.transpose(layer1_op,0,1)
        dim0, dim1, dim2 = layer1_op.shape
        prediction = torch.sum(layer1_op, dim=2).reshape(dim0,dim1)
        return prediction


def load_dataset():
    line_samples = np.load('./2D_dataset/class_0.npy')
    line_targets = np.full(shape=line_samples.shape[0],fill_value=0)
    square_samples = np.load('./2D_dataset/class_2.npy')
    square_targets = np.full(shape=square_samples.shape[0],fill_value=2)
    triangle_samples = np.load('./2D_dataset/class_1.npy')
    triangle_targets = np.full(shape=triangle_samples.shape[0],fill_value=1)

    samples = line_samples
    targets = line_targets
    for sample in [square_samples, triangle_samples]:
        samples = np.vstack((samples,sample))

    for target in [square_targets, triangle_targets]:
        targets = np.hstack((targets,target))

    print(samples.shape)
    print(targets.shape)

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


def custom_loss(y_pred, y, criterion, model):
    hp = 1e-5
    return criterion(y_pred, y) + hp * torch.sum(torch.abs(model.weight1))


def train(model,criterion,optimizer,X,y):
    model.train()
    y_pred = model(X)
    loss = custom_loss(y_pred,y,criterion, model)
    # loss = criterion(y_pred, y)
    loss.backward()  # backprop (compute gradients)
    optimizer.step()  # update weights (gradient descent step)
    optimizer.zero_grad()  # reset gradients
    return loss


def test(model,criterion,X,y,validation):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)  # forward step
        if validation:
            eval_metric = criterion(y_pred, y)  # compute loss
            return eval_metric
        else:
            y_pred = torch.argmax(y_pred,dim=1)
            return y_pred


def main():
    torch.manual_seed(696)
    data = load_dataset()

    X_train, y_train = data['train']['X'], data['train']['y'].type(torch.LongTensor)
    X_validation, y_validation = data['validation']['X'], data['validation']['y'].type(torch.LongTensor)
    X_test, y_test = data['test']['X'], data['test']['y'].type(torch.LongTensor)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lr = 5e-2  # Learning rate

    model = Model2D().to(device)

    criterion = torch.nn.CrossEntropyLoss()  # loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer

    # training the model for 1000 epochs
    optimum_training_loss = float('inf')
    optimum_validation_loss = float('inf')

    for epoch in range(800):
        training_loss = train(model,criterion,optimizer,X_train,y_train)
        validation_loss = test(model,criterion,X_validation,y_validation,validation=True)
        print(f'training loss: {training_loss}, validation loss:{validation_loss}')
        if training_loss < optimum_training_loss and validation_loss < optimum_validation_loss:
            optimum_validation_loss = validation_loss
            optimum_training_loss = training_loss
            torch.save(model.state_dict(), "./models/2D_model" + ".pt")

    model.load_state_dict(torch.load("./models/2D_model" + ".pt"))
    preds = test(model,criterion,X_test,y_test,validation=False)
    print(f'Test accuracy: {accuracy_score(y_test,preds)}')


if __name__ == '__main__':
    main()