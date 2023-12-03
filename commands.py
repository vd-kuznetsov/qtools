import hydra
from omegaconf import DictConfig

from examples.train import train
from examples.infer import infer


@hydra.main(config_path="./configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(cfg)
    if cfg.mode == 'all':
        train(cfg)
        infer(cfg)
    elif cfg.mode == 'train': train(cfg)
    elif cfg.mode == 'infer': infer(cfg)


if __name__ == "__main__":
    main()