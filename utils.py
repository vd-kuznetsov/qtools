import constants

from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


def init_dataloaders(batch_size=constants.BATCH_SIZE):
    # Add hydra config
    data_path = Path('./data/imagewoof')

    # Checking for downloading the dataset from DVC
    if not data_path.exists():
        print('Downloading data from DVC')
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(data_path / 'train', transform=transform)
    val_dataset = datasets.ImageFolder(data_path / 'val', transform=transform)
    calib_dataset = random_split(val_dataset, [140, 20])[1]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    calib_dataloader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dataloader, val_dataloader, calib_dataloader


def model_pipeline(evaluate=False):
    model = models.quantization.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, constants.CLASSES)

    if evaluate:
        return model

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    nn.init.xavier_normal_(model.fc.weight)

    return model
