import torch
import pandas as pd
from torch.nn import functional as F
import logging


# DeepSets Model
class DeepSets(torch.nn.Module):
    def __init__(self,D):
        super(DeepSets, self).__init__()
        self.layer1 = torch.nn.Linear(D,D,bias=False)
        torch.nn.init.kaiming_normal_(self.layer1.weight)

    def forward(self,x):
        layer1_op = self.layer1(x)
        feat_sum = torch.sum(layer1_op,dim=1)
        prediction = F.relu(feat_sum)
        return prediction


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
def train(model, optimizer, criterion, X, y):
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

    lr = 1e-2  # Learning rate

    model = DeepSets(set_size).to(device)

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
    print("Model parameters are: ")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
            print('\nSumming up the final 10x10 weight matrix along dimension=1:')
            print(torch.sum(param.data,dim=0))
            logger.info('\nSumming up the final 10x10 weight matrix along dimension=1:')
            logger.info(torch.sum(param.data,dim=0))


if __name__ == '__main__':
    main()
