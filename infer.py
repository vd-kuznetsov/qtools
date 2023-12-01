import constants
from torch import load, save
from utils import init_dataloader, model_pipeline

from qtools.ptq import ptqs_pipeline


if __name__ == "__main__":
    model = model_pipeline(evaluate=True)
    state_dict = load(constants.MODEL_NAME)
    model.load_state_dict(state_dict)

    train_loader, test_loader = init_dataloader()
    model_quantization = ptqs_pipeline(
        model, test_loader, train_loader, save_predict=True
    )

    if constants.SAVE_QUANT_MODEL:
        save(model_quantization.state_dict(), "resnet18_quant.pth")
