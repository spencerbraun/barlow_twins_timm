import logging
import json
import math
import os
import sys
import time
from datetime import datetime

import torch
import torchvision

import hydra
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

from data import Transform
from model import BarlowTwins
from optim import LARS


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(config, model, loader):

    param_biases = [p for p in model.parameters() if p.ndim == 1]
    param_weights = [p for p in model.parameters() if p.ndim != 1]

    parameters = [{"params": param_weights}, {"params": param_biases}]
    optimizer = LARS(
        parameters,
        lr=0,
        weight_decay=config.weight_decay,
        weight_decay_filter=True,
        lars_adaptation_filter=True,
    )

    # automatically resume from checkpoint if it exists
    if os.path.exists(f"{config.checkpoint_dir}/checkpoint.pth"):
        ckpt = torch.load(f"{config.checkpoint_dir}/checkpoint.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, config.max_epochs):

        for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):

            y1 = y1.to(config.device)
            y2 = y2.to(config.device)

            adjust_learning_rate(config, optimizer, loader, step)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % config.print_freq == 0:

                stats = dict(
                    epoch=epoch,
                    step=step,
                    lr_weights=optimizer.param_groups[0]["lr"],
                    lr_biases=optimizer.param_groups[1]["lr"],
                    loss=loss.item(),
                    time=int(time.time() - start_time),
                )
                print(json.dumps(stats))

        state = dict(
            epoch=epoch + 1,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
        torch.save(state, config.checkpoint_dir / "checkpoint.pth")

    torch.save(
        model.module.encoder.state_dict(), config.checkpoint_dir / "resnet50.pth"
    )


def adjust_learning_rate(config, optimizer, loader, step):
    max_steps = config.max_epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = config.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]["lr"] = lr * config.lr_weights
    optimizer.param_groups[1]["lr"] = lr * config.lr_biases


@hydra.main(config_path="./conf", config_name="config")
def main(config: DictConfig) -> None:

    if config.seed:
        torch.manual_seed(config.seed)

    logger.info(OmegaConf.to_yaml(config, resolve=True))
    logger.info(f"Using the model: {config.model}")

    # dataset = torchvision.datasets.ImageFolder(config.data.path / "train", Transform())
    cwd = get_original_cwd()
    dataset = torchvision.datasets.CIFAR10(
        f"{cwd}/cifar10/", train=True, download=True, transform=Transform()
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.workers,
        pin_memory=True,
    )

    if not config.debug:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{config.run_name}_{config.data.path}_{timestamp}"
        if config.wandb_entity:
            wandb.init(
                entity=config.wandb_entity,
                project=config.wandb_project,
                config=dict(config),
                name=run_name,
            )
            if not config.train.pt:
                config.train.pt = f"{config.train.pt}/{run_name}"

    model = BarlowTwins(config)
    model.to(config.device)

    train(config, model, loader)


if __name__ == "__main__":
    main()
