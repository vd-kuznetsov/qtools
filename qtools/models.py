from torchvision import models
from torchvision.models import quantization


__all__ = [
    "resnet34",
]


def resnet34(weights=models.ResNet34_Weights.DEFAULT):
    model = quantization.resnet.QuantizableResNet(
        quantization.resnet.QuantizableBasicBlock, [3, 4, 6, 3]
    )
    model.load_state_dict(weights.get_state_dict(progress=False))

    return model
