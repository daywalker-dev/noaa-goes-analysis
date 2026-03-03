"""Base trainer with mixed precision, gradient clipping, and checkpointing."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from goes_forecast.training.callbacks import EarlyStopping, CheckpointManager, LRWarmup

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Base training loop with AMP, grad clipping, checkpointing, and logging.

    Subclasses implement _train_step() and _val_step().
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        stage_cfg: DictConfig,
        output_dir: str | Path,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.stage_cfg = stage_cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Optimizer
        opt_cfg = stage_cfg.optimizer
        OptimizerClass = getattr(torch.optim, opt_cfg.type)
        self.optimizer = OptimizerClass(
            [p for p in model.parameters() if p.requires_grad],
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.get("weight_decay", 0),
        )

        # Scheduler
        sched_cfg = stage_cfg.scheduler
        SchedulerClass = getattr(torch.optim.lr_scheduler, sched_cfg.type)
        sched_params = {k: v for k, v in dict(sched_cfg).items() if k != "type"}
        self.scheduler = SchedulerClass(self.optimizer, **sched_params)

        # AMP
        self.use_amp = cfg.training.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Gradient clipping
        self.grad_clip = cfg.training.gradient_clip

        # Callbacks
        ckpt_cfg = cfg.training.checkpointing
        self.checkpoint_mgr = CheckpointManager(
            self.output_dir / "checkpoints",
            save_top_k=ckpt_cfg.save_top_k,
            monitor=ckpt_cfg.monitor,
            mode=ckpt_cfg.mode,
        )
        es_cfg = cfg.training.early_stopping
        self.early_stopping = EarlyStopping(
            patience=es_cfg.patience,
            min_delta=es_cfg.min_delta,
            mode=ckpt_cfg.mode,
        )
        self.warmup = LRWarmup(self.optimizer, warmup_steps=100)

        self.epochs = stage_cfg.epochs
        self.log_interval = cfg.training.logging.log_every_n_steps
        self.global_step = 0

    def _train_step(self, batch: dict) -> dict[str, torch.Tensor]:
        """Implement in subclass. Returns dict of losses."""
        raise NotImplementedError

    def _val_step(self, batch: dict) -> dict[str, torch.Tensor]:
        """Implement in subclass. Returns dict of losses."""
        raise NotImplementedError

    def _to_device(self, batch: dict) -> dict:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_losses: dict[str, list[float]] = {}

        for step, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)
            self.warmup.step()

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                losses = self._train_step(batch)

            total_loss = losses["total"]
            self.scaler.scale(total_loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

            for k, v in losses.items():
                epoch_losses.setdefault(k, []).append(v.item() if torch.is_tensor(v) else v)

            if self.global_step % self.log_interval == 0:
                loss_str = " | ".join(f"{k}: {v.item():.4f}" for k, v in losses.items() if torch.is_tensor(v))
                logger.info(f"[Train] epoch={epoch} step={self.global_step} | {loss_str}")

        return {k: sum(v) / len(v) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        epoch_losses: dict[str, list[float]] = {}

        for batch in self.val_loader:
            batch = self._to_device(batch)
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                losses = self._val_step(batch)

            for k, v in losses.items():
                epoch_losses.setdefault(k, []).append(v.item() if torch.is_tensor(v) else v)

        avg = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in avg.items())
        logger.info(f"[Val] epoch={epoch} | {loss_str}")
        return avg

    def fit(self) -> None:
        """Full training loop."""
        logger.info(f"Starting training: {self.epochs} epochs, device={self.device}")
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            train_losses = self.train_epoch(epoch)
            val_losses = self.validate(epoch)

            self.scheduler.step()

            # Checkpoint
            monitor_key = self.cfg.training.checkpointing.monitor.split("/")[-1]
            val_metric = val_losses.get(monitor_key, val_losses.get("total", 0))
            self.checkpoint_mgr.save(
                self.model.state_dict(), val_metric, epoch,
                extra={"optimizer": self.optimizer.state_dict()},
            )

            # Early stopping
            if self.early_stopping(val_metric):
                break

            elapsed = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch}/{self.epochs} done in {elapsed:.1f}s | "
                f"train_loss={train_losses.get('total', 0):.4f} "
                f"val_loss={val_losses.get('total', 0):.4f}"
            )

        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time / 60:.1f} minutes")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model and optimizer from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        logger.info(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")
