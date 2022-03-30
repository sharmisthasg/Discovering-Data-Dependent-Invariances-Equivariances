from Dataset import Load_dataset
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging
from torchsummary import summary
import itertools
import warnings
warnings.filterwarnings('always')


def calculate_metric(metric_fn, true_y, pred_y):
    try:
        return metric_fn(true_y, pred_y, average="macro")
    except:
        return metric_fn(true_y, pred_y,)


def print_scores(p, r, f1, a, batch_size,logger):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")
        logger.info(f"\t{name.rjust(14, ' ')}: {sum(scores) / batch_size:.4f}")


def f(a):
    return abs(a[0] - a[1])


def custom_loss_fn(criterion, y_pred, y, hp, linear_layers):
    layers = []
    for layer in linear_layers:
        all_pairs = list(itertools.combinations([x.data for x in layer.parameters()][0], 2))
        layers.append(all_pairs)
    reg_sum = 0
    for index in range(len(linear_layers)):
        try:
            reg_sum += hp[index] * sum(sum(list(map(f, layers[index]))))
        except:
            reg_sum += hp[index] * sum(list(map(f, layers[index])))

    return criterion(y_pred, y) + reg_sum


def train(model,criterion,optimizer,train_loader,device,hps,linear_layers):
    model.train()
    total_loss = 0
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        prediction = model(data)
        loss = custom_loss_fn(criterion, prediction, target, hps, linear_layers)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

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
    return precision, recall, f1, accuracy


def main():
    torch.manual_seed(696)
    logging.basicConfig(filename="./reports/three_layers_trial_results.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    train_loader, validation_loader, test_loader = Load_dataset.main()
    for images, labels in train_loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(100*100,64,bias=False),
        nn.ReLU(),
        nn.Linear(64,32,bias=False),
        nn.ReLU(),
        nn.Linear(32,10)
    )

    print(summary(model=model,input_size=(1,100,100),batch_size=256))
    linear_layers = [layer for layer in model.children() if isinstance(layer, nn.Linear)]
    print(linear_layers)

    model.to(device)
    lr = 1e-5
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hps = [1e-6, 1.5e-5, 2e-5]
    optimum_loss = float('inf')

    # training-phase
    print('Training ....')
    for epoch in range(50):
        loss = train(model,criterion,optimizer,train_loader,device,hps,linear_layers)
        print(f'Epoch {epoch}: Training Loss {loss}')
        logger.info(f'Epoch {epoch}: Training Loss {loss}')
        if loss < optimum_loss:
            optimum_loss = loss
            torch.save(model.state_dict(), "models/three_layers_trial" + ".pt")

    # test-phase
    print('Testing ...')
    print('\nRESULTS ON TEST DATA:')
    logger.info('\nRESULTS ON TEST DATA:')
    model.load_state_dict(torch.load("models/three_layers_trial" + ".pt"))
    precision, recall, f1, accuracy = test(model, test_loader, device)
    print_scores(precision, recall, f1, accuracy, len(test_loader),logger)


if __name__ == '__main__':
    main()