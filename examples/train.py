import subprocess

import mlflow
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim, save

from qtools.metrics import AverageMeter, accuracy

from .utils import init_dataloaders, model_pipeline


def train(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    experiment = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)

    if experiment is None:
        mlflow.create_experiment(cfg.mlflow.experiment_name)

    train_dataloader, _ = init_dataloaders(cfg)
    model = model_pipeline(cfg)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    print("Start training")
    train_model(cfg, model, optimizer, criterion, train_dataloader)
    print("Stop training")


def train_model(cfg: DictConfig, model, optimizer, criterion, train_dataloader):
    experiment_id = mlflow.get_experiment_by_name(
        cfg.mlflow.experiment_name
    ).experiment_id
    with mlflow.start_run(experiment_id=experiment_id):
        hydra_log_params = OmegaConf.to_container(cfg, resolve=True)
        hydra_log_params["git_commit_id"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        )
        mlflow.log_params(hydra_log_params)
        for epoch in range(cfg.training.epochs):
            model.train()
            top1 = AverageMeter("Acc@1", ":6.2f")
            top5 = AverageMeter("Acc@5", ":6.2f")
            avg_loss = 0.0

            for batch in train_dataloader:
                inputs, labels = batch
                optimizer.zero_grad()
                outputs = model(inputs)
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()

            mlflow.log_metric("Train Accuracy_1", top1.avg, step=epoch)
            mlflow.log_metric("Train Accuracy_5", top5.avg, step=epoch)
            mlflow.log_metric("Train Loss", avg_loss / len(train_dataloader), step=epoch)

    save(model.state_dict(), cfg.model.name + "_no_quant.pth")


if __name__ == "__main__":
    raise RuntimeError("Use `python commands.py mode=train`")
