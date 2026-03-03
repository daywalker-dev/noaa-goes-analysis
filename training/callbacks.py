"""Training callbacks: LR warmup, early stopping, checkpointing, logging."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when monitored metric stops improving."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def __call__(self, metric: float) -> bool:
        improved = (
            metric < self.best - self.min_delta if self.mode == "min"
            else metric > self.best + self.min_delta
        )
        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
            return True
        return False


class CheckpointManager:
    """Save top-k checkpoints based on a monitored metric."""

    def __init__(
        self,
        save_dir: str | Path,
        save_top_k: int = 3,
        monitor: str = "val/loss",
        mode: str = "min",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.checkpoints: list[tuple[float, Path]] = []

    def save(
        self,
        state_dict: dict,
        metric: float,
        epoch: int,
        extra: Optional[dict] = None,
    ) -> Optional[Path]:
        """Save checkpoint if metric is in top-k.

        Returns:
            Path to saved checkpoint, or None if not saved.
        """
        payload = {"state_dict": state_dict, "epoch": epoch, "metric": metric}
        if extra:
            payload.update(extra)

        # Check if this qualifies for top-k
        if len(self.checkpoints) < self.save_top_k or self._is_better(metric):
            path = self.save_dir / f"epoch_{epoch:04d}_metric_{metric:.6f}.ckpt"
            torch.save(payload, path)
            self.checkpoints.append((metric, path))
            self.checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

            # Remove worst if over limit
            while len(self.checkpoints) > self.save_top_k:
                _, old_path = self.checkpoints.pop()
                if old_path.exists():
                    old_path.unlink()

            # Symlink best
            best_link = self.save_dir / "best.ckpt"
            if best_link.exists() or best_link.is_symlink():
                best_link.unlink()
            best_link.symlink_to(self.checkpoints[0][1].name)

            logger.info(f"Saved checkpoint: {path.name} (metric={metric:.6f})")
            return path
        return None

    def _is_better(self, metric: float) -> bool:
        worst = self.checkpoints[-1][0] if self.checkpoints else None
        if worst is None:
            return True
        return metric < worst if self.mode == "min" else metric > worst


class LRWarmup:
    """Linear learning rate warmup for the first N steps."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int = 1000):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / self.warmup_steps
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base_lr * scale
