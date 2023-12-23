import hydra
from examples.export_model import export_model
from examples.infer import infer
from examples.train import train
from omegaconf import DictConfig


@hydra.main(config_path="./configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(cfg)
    if cfg.mode == "all":
        train(cfg)
        infer(cfg)
    elif cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "infer":
        infer(cfg)
    elif cfg.mode == "export":
        export_model(cfg)


if __name__ == "__main__":
    main()
