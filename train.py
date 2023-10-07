import constants
from torch import nn, optim, save
from utils import init_dataloader, model_pipeline


def train(model, optimizer, criterion, train_loader):
    for _ in range(constants.EPOCH):
        model.train()
        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    save(model.state_dict(), constants.MODEL_NAME)


if __name__ == "__main__":
    train_loader, _ = init_dataloader()
    model = model_pipeline()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=constants.LR)
    train(model, optimizer, criterion, train_loader)
