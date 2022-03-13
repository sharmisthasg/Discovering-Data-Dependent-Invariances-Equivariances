from Dataset import Load_subset_dataset
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging
from torchsummary import summary
import itertools
import warnings
warnings.filterwarnings('always')
from collections import OrderedDict


def calculate_metric(metric_fn, true_y, pred_y):
    return metric_fn(true_y, pred_y,)


def print_scores(p, r, f1, a, batch_size,logger):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(name, scores)


def f(a):
    return torch.abs(a[0] - a[1])


def custom_loss_fn(criterion, y_pred, y, hp, model):
    all_pairs = list(itertools.combinations([x.data for x in model.linear.weight], 2))
    return criterion(y_pred, y) + hp*sum(sum(list(map(f, all_pairs))))


def train(model,criterion,optimizer,train_loader,device,hps):
    model.train()
    total_loss = 0
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        prediction = model(data)
        loss = custom_loss_fn(criterion, prediction, target, hps, model)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def validation(model,criterion,validation_loader,device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            prediction = model(data)
            loss = criterion(prediction, target)
            total_loss += loss

    return total_loss


def test(model,test_loader,device):
    model.eval()
    precision, recall, f1, accuracy = [], [], [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions = output.argmax(dim=1, keepdim=True)
            for acc, metric in zip((precision, recall, f1, accuracy),
                                   (precision_score, recall_score, f1_score, accuracy_score)):
                acc.append(
                    calculate_metric(metric, target.cpu(), predictions.cpu())
                )
            break
    return precision, recall, f1, accuracy


def main():
    torch.manual_seed(696)
    logging.basicConfig(filename="reports/1_layer_0-1_MLP.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    train_loader, validation_loader, test_loader = Load_subset_dataset.main()
    for images, labels in train_loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(OrderedDict([
        ('flatten', nn.Flatten()),
        ('linear', nn.Linear(10*10,2,bias=False))
    ])
    )

    print(summary(model=model,input_size=(1,10,10),batch_size=128))
    # print([sum(x.data) for x in model.linear.weight])

    model.to(device)
    lr = 2e-6
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hps = 1e-6
    optimum_loss = float('inf')
    optimum_val_loss = float('inf')

    # training-phase
    print('Training ....')
    for epoch in range(15):
        loss = train(model,criterion,optimizer,train_loader,device,hps)
        val_loss = validation(model, criterion, validation_loader, device)
        print(f'Epoch {epoch}: Training Loss {loss} Validation Loss {val_loss}')
        logger.info(f'Epoch {epoch}: Training Loss {loss} Validation Loss {val_loss}')
        if loss < optimum_loss and val_loss < optimum_val_loss :
            optimum_loss = loss
            optimum_val_loss = val_loss
            print('updating saved model')
            torch.save(model.state_dict(), "models/1_layer_0-1_MLP" + ".pt")

    # test-phase
    print('Testing ...')
    print('\nRESULTS ON TEST DATA:')
    logger.info('\nRESULTS ON TEST DATA:')
    model.load_state_dict(torch.load("models/1_layer_0-1_MLP" + ".pt"))
    precision, recall, f1, accuracy = test(model, test_loader, device)
    print_scores(precision, recall, f1, accuracy, len(test_loader),logger)


if __name__ == '__main__':
    main()