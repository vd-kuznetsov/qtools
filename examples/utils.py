from pathlib import Path

from dvc.api import DVCFileSystem
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def init_dataloaders(cfg: DictConfig):
    data_path = Path("./" + cfg.data.path + "/" + cfg.data.name)

    if not data_path.exists():
        print("Downloading data from DVC")
        fs = DVCFileSystem(".", subrepos=True, rev=cfg.data.rev_dvc)
        fs.get(cfg.data.path, cfg.data.path, recursive=True)

    transform = transforms.Compose(
        [
            transforms.Resize(cfg.transforms.resize),
            transforms.CenterCrop(cfg.transforms.center_crop),
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.ImageFolder(
        data_path / cfg.data.train_path, transform=transform
    )
    val_dataset = datasets.ImageFolder(data_path / cfg.data.val_path, transform=transform)

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.training.batch_size, shuffle=False, drop_last=True
    )

    return train_dataloader, val_dataloader


def model_pipeline(cfg: DictConfig, evaluate=False):
    model = models.quantization.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, cfg.model.classes)

    if evaluate:
        return model

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    nn.init.xavier_normal_(model.fc.weight)
    return model
