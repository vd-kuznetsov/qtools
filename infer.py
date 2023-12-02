import constants
from qtools.ptq import ptqs_pipeline
from torch import load, save
from utils import init_dataloaders, model_pipeline


if __name__ == "__main__":
    model = model_pipeline(evaluate=True)
    state_dict = load(constants.MODEL_NAME)
    model.load_state_dict(state_dict)

    _, val_dataloader, calib_dataloader = init_dataloaders()
    
    model_quantization = ptqs_pipeline(
        model, val_dataloader, calib_dataloader, save_predict=True
    )

    if constants.SAVE_QUANT_MODEL:
        save(model_quantization.state_dict(), "resnet18_quant.pth")
