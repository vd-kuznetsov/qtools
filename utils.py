import constants
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def init_dataloader(batch_size=constants.BATCH_SIZE):
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.FashionMNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        "./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def model_pipeline(evaluate=False):
    model = models.quantization.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, constants.CLASSES)

    # relevant for data from FashionMNIST
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False
    )

    if evaluate:
        return model

    for param in model.parameters():
        param.requires_grad = False

    for param in model.conv1.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    nn.init.xavier_normal_(model.fc.weight)
    nn.init.xavier_normal_(model.conv1.weight)

    return model
