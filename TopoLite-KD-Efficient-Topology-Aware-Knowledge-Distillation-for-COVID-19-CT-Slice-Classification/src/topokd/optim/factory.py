from __future__ import annotations

import math

import torch

from .sam import SAM


def build_optimizer(model: torch.nn.Module, config: dict):
    cfg = config["optimizer"]
    name = cfg["name"].lower()
    kwargs = {
        "lr": float(cfg["learning_rate"]),
        "weight_decay": float(cfg["weight_decay"]),
        "betas": tuple(float(value) for value in cfg.get("betas", [0.9, 0.999])),
    }
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(parameters, **kwargs)
    if name == "sam":
        return SAM(
            parameters,
            torch.optim.AdamW,
            rho=float(cfg.get("sam_rho", 0.05)),
            adaptive=bool(cfg.get("sam_adaptive", False)),
            **kwargs,
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=kwargs["lr"],
            weight_decay=kwargs["weight_decay"],
            momentum=0.9,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, config: dict, steps_per_epoch: int):
    cfg = config["scheduler"]
    name = cfg["name"].lower()
    epochs = int(config["training"]["epochs"])
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    min_lr = float(cfg.get("min_learning_rate", 0.0))
    base_lr = float(config["optimizer"]["learning_rate"])
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(0, warmup_epochs * steps_per_epoch)

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        if name == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
            return (min_lr / base_lr) + (1.0 - min_lr / base_lr) * cosine
        if name == "constant":
            return 1.0
        raise ValueError(f"Unsupported scheduler: {name}")

    target_optimizer = optimizer.base_optimizer if isinstance(optimizer, SAM) else optimizer
    return torch.optim.lr_scheduler.LambdaLR(target_optimizer, lr_lambda=multiplier)
