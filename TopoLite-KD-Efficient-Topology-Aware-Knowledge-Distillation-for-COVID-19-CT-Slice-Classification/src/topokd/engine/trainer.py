from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, *_args, **_kwargs):
            pass
        def add_scalar(self, *_args, **_kwargs):
            pass
        def close(self):
            pass
from tqdm.auto import tqdm

from topokd.config import save_resolved_config
from topokd.evaluation.metrics import binary_metrics
from topokd.losses import ResearchObjective
from topokd.optim import SAM, build_optimizer, build_scheduler
from topokd.utils.checkpoint import load_checkpoint, save_checkpoint
from topokd.utils.environment import capture_environment
from topokd.utils.io import append_csv, ensure_dir
from topokd.utils.logging import create_logger
from topokd.visualization import generate_gradcams, plot_training_history

from .evaluator import evaluate_splits
from .inference import model_inputs


class Trainer:
    def __init__(self, config: dict, student, teacher, loaders: dict, device: torch.device):
        self.config = config
        self.student = student.to(device)
        self.teacher = teacher.to(device) if teacher is not None else None
        self.loaders = loaders
        self.device = device
        self.objective = ResearchObjective(config, device)
        self.optimizer = build_optimizer(self.student, config)
        self.scheduler = build_scheduler(self.optimizer, config, len(loaders["train"]))
        seed = int(config["training"]["seed"])
        experiment = config["project"]["experiment_name"]
        planned_run_dir = Path(config["project"]["output_root"]) / experiment / f"seed_{seed}"
        if planned_run_dir.exists() and any(planned_run_dir.iterdir()) and not config["training"].get("resume"):
            raise FileExistsError(
                f"Run directory already contains artifacts: {planned_run_dir}. Use a new seed/experiment name or set training.resume."
            )
        self.run_dir = ensure_dir(planned_run_dir)
        self.checkpoint_dir = ensure_dir(self.run_dir / "checkpoints")
        self.log_dir = ensure_dir(self.run_dir / "logs")
        self.history_path = self.log_dir / "history.csv"
        self.logger = create_logger(experiment, self.log_dir / "run.log")
        self.writer = SummaryWriter(self.log_dir / "tensorboard")
        save_resolved_config(config, self.run_dir / "resolved_config.yaml")
        capture_environment(self.run_dir / "logs")
        self.start_epoch = 1
        self.best_score = -math.inf if config["training"]["monitor_mode"] == "max" else math.inf
        self.bad_epochs = 0
        self.scaler = torch.amp.GradScaler("cuda", enabled=bool(config["training"].get("amp", True)) and device.type == "cuda")
        if config["training"].get("resume"):
            self._resume(config["training"]["resume"])

    def _resume(self, path: str) -> None:
        payload = load_checkpoint(path, self.device)
        self.student.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if payload.get("scheduler_state"):
            self.scheduler.load_state_dict(payload["scheduler_state"])
        self.start_epoch = int(payload["epoch"]) + 1
        self.best_score = float(payload.get("best_score", self.best_score))
        self.logger.info("Resumed from %s at epoch %d", path, self.start_epoch)

    def fit(self) -> dict:
        epochs = int(self.config["training"]["epochs"])
        patience = int(self.config["training"].get("early_stopping_patience", 0))
        for epoch in range(self.start_epoch, epochs + 1):
            started = time.perf_counter()
            train_loss, train_metrics = self._train_epoch(epoch)
            val_loss, val_metrics = self._validation_epoch(epoch)
            current_lr = self._learning_rate()
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
                **{f"train_{key}": value for key, value in train_metrics.items() if key not in {"threshold", "n", "tn", "fp", "fn", "tp"}},
                **{f"val_{key}": value for key, value in val_metrics.items() if key not in {"threshold", "n", "tn", "fp", "fn", "tp"}},
                "epoch_seconds": time.perf_counter() - started,
            }
            append_csv(self.history_path, row)
            for key, value in row.items():
                if isinstance(value, (float, int)) and key != "epoch":
                    self.writer.add_scalar(key, value, epoch)
            monitor_key = self.config["training"]["monitor"]
            score = float(row[monitor_key])
            improved = self._is_improved(score)
            self._save_epoch(epoch, score, improved)
            if improved:
                self.best_score = score
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
            self.logger.info(
                "Epoch %03d | train_loss %.4f | val_loss %.4f | val_mcc %.4f | val_auroc %.4f | best %.4f",
                epoch,
                train_loss,
                val_loss,
                val_metrics["mcc"],
                val_metrics["auroc"],
                self.best_score,
            )
            if patience > 0 and self.bad_epochs >= patience:
                self.logger.info("Early stopping after %d non-improving epochs.", self.bad_epochs)
                break
        best = load_checkpoint(self.checkpoint_dir / "best.pt", self.device)
        self.student.load_state_dict(best["model_state"])
        results = evaluate_splits(
            self.student,
            self.loaders["val"],
            self.loaders["test"],
            self.config,
            self.device,
            self.run_dir,
        )
        plot_training_history(self.history_path, self.run_dir / "figures")
        gradcam_count = int(self.config["evaluation"].get("gradcam_samples_per_class", 0))
        if gradcam_count > 0 and hasattr(self.student, "visual_encoder"):
            generate_gradcams(
                model=self.student,
                dataset=self.loaders["test"].dataset,
                predictions_csv=self.run_dir / "predictions" / "test_predictions.csv",
                output_dir=self.run_dir / "gradcam",
                target_layer=self.config["evaluation"]["target_layer"],
                threshold=float(results["calibration"]["threshold"]),
                per_class=gradcam_count,
                device=self.device,
            )
        self.writer.close()
        return results

    def _train_epoch(self, epoch: int) -> tuple[float, dict]:
        self.student.train()
        if self.teacher is not None:
            self.teacher.eval()
        losses = []
        labels_all, probabilities_all = [], []
        progress = tqdm(self.loaders["train"], desc=f"Train {epoch}", leave=False)
        for batch in progress:
            labels = batch["label"].to(self.device, non_blocking=True)
            kwargs = model_inputs(batch, self.device)
            if isinstance(self.optimizer, SAM):
                loss, output = self._sam_step(kwargs, labels)
            else:
                loss, output = self._standard_step(kwargs, labels)
            losses.append(float(loss))
            labels_all.append(labels.detach().cpu().numpy())
            probabilities_all.append(torch.sigmoid(output["logits"].detach()).cpu().numpy())
            progress.set_postfix(loss=f"{loss:.4f}")
        labels_np = np.concatenate(labels_all)
        probabilities_np = np.concatenate(probabilities_all)
        return float(np.mean(losses)), binary_metrics(labels_np, probabilities_np, 0.5, int(self.config["evaluation"]["ece_bins"]))

    def _teacher_forward(self, image: torch.Tensor):
        if self.teacher is None:
            return None
        with torch.inference_mode():
            return self.teacher(image)

    def _standard_step(self, kwargs: dict, labels: torch.Tensor) -> tuple[float, dict]:
        self.optimizer.zero_grad(set_to_none=True)
        amp_enabled = self.scaler.is_enabled()
        with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
            output = self.student(**kwargs)
            teacher_output = self._teacher_forward(kwargs["image"])
            loss, _ = self.objective(output, labels, teacher_output)
        self.scaler.scale(loss).backward()
        clip = float(self.config["training"].get("gradient_clip_norm", 0.0))
        if clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return float(loss.detach()), output

    def _sam_step(self, kwargs: dict, labels: torch.Tensor) -> tuple[float, dict]:
        self.optimizer.zero_grad(set_to_none=True)
        teacher_output = self._teacher_forward(kwargs["image"])
        output = self.student(**kwargs)
        loss, _ = self.objective(output, labels, teacher_output)
        loss.backward()
        clip = float(self.config["training"].get("gradient_clip_norm", 0.0))
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), clip)
        self.optimizer.first_step(zero_grad=True)
        second_output = self.student(**kwargs)
        second_loss, _ = self.objective(second_output, labels, teacher_output)
        second_loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), clip)
        self.optimizer.second_step(zero_grad=True)
        self.scheduler.step()
        return float(loss.detach()), output

    @torch.inference_mode()
    def _validation_epoch(self, epoch: int) -> tuple[float, dict]:
        self.student.eval()
        if self.teacher is not None:
            self.teacher.eval()
        losses, labels_all, probabilities_all = [], [], []
        for batch in tqdm(self.loaders["val"], desc=f"Validate {epoch}", leave=False):
            labels = batch["label"].to(self.device, non_blocking=True)
            kwargs = model_inputs(batch, self.device)
            output = self.student(**kwargs)
            teacher_output = self._teacher_forward(kwargs["image"])
            loss, _ = self.objective(output, labels, teacher_output)
            losses.append(float(loss))
            labels_all.append(labels.cpu().numpy())
            probabilities_all.append(torch.sigmoid(output["logits"]).cpu().numpy())
        labels_np = np.concatenate(labels_all)
        probabilities_np = np.concatenate(probabilities_all)
        return float(np.mean(losses)), binary_metrics(labels_np, probabilities_np, 0.5, int(self.config["evaluation"]["ece_bins"]))

    def _is_improved(self, score: float) -> bool:
        return score > self.best_score if self.config["training"]["monitor_mode"] == "max" else score < self.best_score

    def _save_epoch(self, epoch: int, score: float, improved: bool) -> None:
        payload = {
            "epoch": epoch,
            "best_score": self.best_score if not improved else score,
            "model_state": self.student.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.config,
        }
        save_checkpoint(self.checkpoint_dir / "last.pt", **payload)
        if improved:
            save_checkpoint(self.checkpoint_dir / "best.pt", **payload)
        frequency = int(self.config["training"].get("save_every_epochs", 0))
        if frequency > 0 and epoch % frequency == 0:
            save_checkpoint(self.checkpoint_dir / f"epoch_{epoch:03d}.pt", **payload)

    def _learning_rate(self) -> float:
        optimizer = self.optimizer.base_optimizer if isinstance(self.optimizer, SAM) else self.optimizer
        return float(optimizer.param_groups[0]["lr"])
