from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topokd.config import apply_overrides, load_config, save_resolved_config
from topokd.data import build_dataloaders
from topokd.evaluation.metrics import binary_metrics
from topokd.models.teacher import EfficientNetTeacher
from topokd.utils import resolve_device, seed_everything
from topokd.utils.checkpoint import save_checkpoint
from topokd.utils.environment import capture_environment
from topokd.utils.io import append_csv, atomic_json_dump, ensure_dir
from topokd.visualization import plot_training_history


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    losses, labels_all, probabilities_all = [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)["logits"]
        losses.append(float(F.binary_cross_entropy_with_logits(logits, labels)))
        labels_all.append(labels.cpu().numpy())
        probabilities_all.append(torch.sigmoid(logits).cpu().numpy())
    labels = np.concatenate(labels_all)
    probabilities = np.concatenate(probabilities_all)
    return float(np.mean(losses)), binary_metrics(labels, probabilities, 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the EfficientNet-B0 teacher on the frozen manifest.")
    parser.add_argument("--config", default="configs/topolite_kd.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    teacher_config = copy.deepcopy(config)
    teacher_config["topology"]["enabled"] = False
    seed = int(config["training"]["seed"])
    seed_everything(seed, bool(config["training"].get("deterministic", True)))
    device = resolve_device(args.device)
    loaders = build_dataloaders(teacher_config)
    teacher_cfg = config["model"]["teacher"]
    model = EfficientNetTeacher(
        pretrained=bool(teacher_cfg.get("pretrained", True)),
        checkpoint=None,
        freeze=False,
        backbone_checkpoint=teacher_cfg.get("backbone_checkpoint"),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    checkpoint_path = Path(teacher_cfg["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = ensure_dir(Path(config["project"]["output_root"]) / "teacher_efficientnet_b0" / f"seed_{seed}")
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "metrics")
    ensure_dir(run_dir / "figures")
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    capture_environment(run_dir / "logs")
    history_path = run_dir / "logs" / "history.csv"

    best_mcc = -1.0
    best_metrics: dict = {}
    patience = 8
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(loaders["train"], desc=f"Teacher {epoch}", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)["logits"]
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        val_loss, metrics = evaluate(model, loaders["val"], device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"val_{key}": value for key, value in metrics.items() if key not in {"threshold", "n", "tn", "fp", "fn", "tp"}},
        }
        append_csv(history_path, row)
        print(f"Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} | val_loss={val_loss:.4f} | val_mcc={metrics['mcc']:.4f}")
        if metrics["mcc"] > best_mcc:
            best_mcc = float(metrics["mcc"])
            best_metrics = metrics
            bad_epochs = 0
            payload = dict(
                epoch=epoch,
                best_mcc=best_mcc,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                config=config,
            )
            save_checkpoint(checkpoint_path, **payload)
            save_checkpoint(run_dir / "checkpoints" / "best.pt", **payload)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    atomic_json_dump(best_metrics, run_dir / "metrics" / "best_validation_metrics.json")
    plot_training_history(history_path, run_dir / "figures")
    (run_dir / "teacher_summary.json").write_text(
        json.dumps({"checkpoint": str(checkpoint_path.resolve()), "best_validation_mcc": best_mcc}, indent=2),
        encoding="utf-8",
    )
    print(f"Best teacher checkpoint: {checkpoint_path.resolve()} | validation MCC={best_mcc:.4f}")
    print(f"Teacher run directory: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
