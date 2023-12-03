from omegaconf import DictConfig
from torch import nn, optim, save
from .utils import init_dataloaders, model_pipeline


def train(cfg: DictConfig):
    train_dataloader, _ = init_dataloaders(cfg)
    model = model_pipeline(cfg)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    print("Start training")
    train_model(cfg, model, optimizer, criterion, train_dataloader)
    print("Stop training")


def train_model(cfg: DictConfig, model, optimizer, criterion, train_dataloader):
    for _ in range(cfg.training.epochs):
        model.train()
        for batch in train_dataloader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    save(model.state_dict(), cfg.model.name + "_no_quant.pth")


if __name__ == "__main__":
    raise RuntimeError("Use `python commands.py mode=train`")
