from Dataset import Load_dataset
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging
import warnings
warnings.filterwarnings('always')


def calculate_metric(metric_fn, true_y, pred_y):
    try:
        return metric_fn(true_y, pred_y, average="macro")
    except:
        return metric_fn(true_y, pred_y,)


def print_scores(p, r, f1, a, batch_size, logger):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")
        logger.info(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")


def train(model,criterion,optimizer,train_loader,device):
    model.train()
    total_loss = 0
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        prediction = model(data)
        loss = criterion(prediction, target)
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
    logging.basicConfig(filename="./reports/baseline.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    train_loader, validation_loader, test_loader = Load_dataset.main()
    logger.info(f"Train loader size: {len(train_loader)}, Validation loader size: {len(validation_loader)}, Test loader size: {len(test_loader)}")

    for images, labels in train_loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  #only 1 channel greyscale images
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    lr = 1e-2
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training-phase
    optimum_loss = float('inf')
    for epoch in range(10):
        loss = train(model,criterion,optimizer,train_loader,device)
        print(f'Epoch {epoch}: Training Loss {loss}')
        logger.info(f'Epoch {epoch}: Training Loss {loss}')
        if loss < optimum_loss:
            optimum_loss = loss
            torch.save(model.state_dict(), "models/baseline" + ".pt")

    # test-phase
    print('\nRESULTS ON TEST DATA:')
    logger.info('\nRESULTS ON TEST DATA:')
    model.load_state_dict(torch.load("models/baseline" + ".pt"))
    precision, recall, f1, accuracy = test(model,test_loader,device)
    print_scores(precision, recall, f1, accuracy, len(test_loader), logger)


if __name__ == '__main__':
    main()