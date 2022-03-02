import torch
import pandas as pd
from torch.nn import functional as F
import logging


# DeepSets Model
class DeepSets(torch.nn.Module):
    def __init__(self):
        super(DeepSets, self).__init__()
        self.l1 = torch.nn.Linear(1, 10)
        torch.nn.init.kaiming_normal_(self.l1.weight)
        self.bn1 = torch.nn.BatchNorm1d(10)
        self.l2 = torch.nn.Linear(10, 1)
        torch.nn.init.kaiming_normal_(self.l2.weight)
        self.bn2 = torch.nn.BatchNorm1d(1)

    def forward(self,x):
        x_expanded = x.reshape((x.shape[0]*x.shape[1],1))
        features = F.relu(self.bn1(self.l1(x_expanded)))
        features = F.relu(self.bn2(self.l2(features)))
        features = features.reshape((x.shape[0], x.shape[1]))
        feat_sum = torch.sum(features, dim=1)
        feat_sum = feat_sum.reshape((feat_sum.shape[0], 1))
        y_pred = F.relu(feat_sum)
        return y_pred


# loading dataset
def load_data(data_dir,set_size):
    X_train = pd.read_csv(f'{data_dir}/Training.csv', header=None).values[:,:set_size]
    y_train = pd.read_csv(f'{data_dir}/Training.csv', header=None).values[:, set_size]

    X_test = pd.read_csv(f'{data_dir}/Test.csv', header=None).values[:,:set_size]
    y_test = pd.read_csv(f'{data_dir}/Test.csv', header=None).values[:, set_size]

    X_validation = pd.read_csv(f'{data_dir}/Validation.csv', header=None).values[:,:set_size]
    y_validation = pd.read_csv(f'{data_dir}/Validation.csv', header=None).values[:, set_size]

    return {"train": dict(X=torch.Tensor(X_train), y=torch.Tensor(y_train)),
            "test": dict(X=torch.Tensor(X_test), y=torch.Tensor(y_test)),
            "validation": dict(X=torch.Tensor(X_validation), y=torch.Tensor(y_validation))}


# train module
def train(model,optimizer, criterion, X, y):
    y_pred = model(X).reshape(y.size())  # forward step
    loss = criterion(y_pred,y)
    loss.backward()  # backprop (compute gradients)
    optimizer.step()  # update weights (gradient descent step)
    optimizer.zero_grad()  # reset gradients
    return loss


# test module
def test(model, X, y, criterion):
    model.eval()
    with torch.no_grad():
        y_pred = model(X).reshape(y.size())  # forward step
        eval_metric = criterion(y_pred,y)  # compute loss
        return eval_metric


def main():
    logging.basicConfig(filename="./reports/baseline.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    set_size = 10
    data = load_data('./Dataset',set_size)

    X_train = data['train']['X']
    y_train = data['train']['y']
    print(f"Training data loaded: X shape: {X_train.size()}, y shape: {y_train.size()}")
    logger.info(f"Training data loaded: X shape: {X_train.size()}, y shape: {y_train.size()}")

    X_test = data['test']['X']
    y_test = data['test']['y']
    print(f"Test data loaded: X shape: {X_test.size()}, y shape: {y_test.size()}")
    logger.info(f"Test data loaded: X shape: {X_test.size()}, y shape: {y_test.size()}")

    X_validation = data['validation']['X']
    y_validation = data['validation']['y']
    print(f"Validation data loaded: X shape: {X_validation.size()}, y shape: {y_validation.size()}")
    logger.info(f"Validation data loaded: X shape: {X_validation.size()}, y shape: {y_validation.size()}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lr = 1e-2 # Learning rate

    model = DeepSets().to(device)

    criterion = torch.nn.L1Loss(reduction='mean')  # loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer

    # training the model for 1000 epochs
    model.train()

    optimum_training_loss = float('inf')
    for epoch in range(1000):
        training_loss = train(model,optimizer,criterion,X_train,y_train)
        if epoch % 50 == 0:
            print(f"[EPOCH]: {epoch}, [TRAINING_LOSS]: {training_loss.item():.6f}")
            logger.info(f"[EPOCH]: {epoch}, [TRAINING_LOSS]: {training_loss.item():.6f}")
        if training_loss.item() < optimum_training_loss:
            torch.save(model.state_dict(), "models/baseline" + ".pt")
            optimum_training_loss = training_loss

    # test-phase
    model.load_state_dict(torch.load("models/baseline" + ".pt"))
    evaluation = test(model, X_test, y_test, criterion)
    print(f"\n[TEST_LOSS]: {evaluation.item():.6f}")
    logger.info(f"\n[TEST_LOSS]: {evaluation.item():.6f}")


if __name__ == '__main__':
    main()
