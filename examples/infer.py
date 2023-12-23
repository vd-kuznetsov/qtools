from omegaconf import DictConfig
from torch import load, save

from qtools.ptq import ptqs_pipeline

from .utils import init_dataloaders, model_pipeline


def infer(cfg: DictConfig):
    model = model_pipeline(cfg, evaluate=True)
    state_dict = load(cfg.model.name + "_no_quant.pth")
    model.load_state_dict(state_dict)

    train_dataloader, val_dataloader = init_dataloaders(cfg)

    model_quantization = ptqs_pipeline(
        model, val_dataloader, train_dataloader, save_predict=False, tag="eager"
    )

    if cfg.model.save_quant_model:
        save(model_quantization.state_dict(), cfg.model.name + "_quant.pth")


if __name__ == "__main__":
    raise RuntimeError("Use `python commands.py mode=infer`")
