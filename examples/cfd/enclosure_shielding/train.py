# train.py
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Training Script for Time-Conditional GeoTransolver E-Field Magnitude Prediction.

During training, a random timestep is sampled per iteration.
The model predicts [N, 1] at that timestep, and the loss is computed
against the corresponding target slice.

During validation, a full rollout produces [N, T, 1].
"""

import os
import sys
import time
import logging


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


import hydra
import omegaconf
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

import torch
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils import load_checkpoint, save_checkpoint

from datapipe import SimSample, simsample_collate

# Optional: tabulate for metrics tables, torchinfo for model summary
_tabulate = OptionalImport("tabulate")
_torchinfo = OptionalImport("torchinfo")


# ═══════════════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Trainer for time-conditional GeoTransolver |E| prediction.

    Data flow (training):
        Input:  coords [N,3], features [N,1], geometry [M,3], time (scalar)
        Model:  single forward pass at random t -> pred [N, 1]
        Target: |E| at timestep t, shape [N, 1]
        Loss:   MSE over [N, 1]

    Data flow (validation):
        Model:  full rollout -> pred [N, T, 1]
        Target: full sequence [N, T, 1]
        Loss:   MSE over [N, T, 1]
    """

    def __init__(self, cfg: DictConfig, logger0: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.cfg = cfg
        self.rollout_steps = cfg.training.num_time_steps - 1
        self.amp = cfg.training.amp

        # ── Dataset ──
        reader = instantiate(cfg.reader)
        logging.getLogger().setLevel(logging.INFO)

        dataset = instantiate(
            cfg.datapipe,
            name="emag_train",
            reader=reader,
            split="train",
            logger=logger0,
        )
        logging.getLogger().setLevel(logging.INFO)

        # Move stats to device
        self.data_stats = dict(
            node={k: v.to(self.dist.device) for k, v in dataset.node_stats.items()},
            feature={
                k: v.to(self.dist.device)
                for k, v in dataset.feature_stats.items()
            },
            geometry={
                k: v.to(self.dist.device)
                for k, v in dataset.geometry_stats.items()
            },
        )

        # Sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.dist.world_size,
            rank=self.dist.rank,
            shuffle=True,
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=cfg.training.num_dataloader_workers,
            sampler=sampler,
            collate_fn=simsample_collate,
        )
        self.sampler = sampler

        # ── Validation ──
        self.val_dataloader = None
        self.num_validation_samples = 0
        self.num_validation_replicas = 0
        if cfg.training.num_validation_samples > 0:
            self._setup_validation(cfg, reader, logger0)

        # ── Model ──
        self.model = instantiate(cfg.model)
        logging.getLogger().setLevel(logging.INFO)
        self.model.to(self.dist.device)
        self.model.train()

        # Log model summary and parameter count
        if self.dist.rank == 0:
            num_params = sum(p.numel() for p in self.model.parameters())
            logger0.info(f"Model parameters: {num_params:,}")
            if _torchinfo.available:
                try:
                    logger0.info(f"\n{_torchinfo.summary(self.model, verbose=0)}")
                except Exception:
                    logger0.info(
                        "(torchinfo summary skipped: model requires sample input)"
                    )

        # Distributed data parallel
        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        # ── Loss ──
        self.criterion = torch.nn.MSELoss()

        # ── Optimizer ──
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.start_lr,
            weight_decay=cfg.training.get("weight_decay", 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        logger0.info(f"Using {self.optimizer.__class__.__name__} optimizer")

        # ── Scheduler ──
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.epochs,
            eta_min=cfg.training.get("min_lr", 1e-6),
        )
        self.scaler = GradScaler("cuda", enabled=self.amp)

        # ── Checkpoint ──
        if self.dist.world_size > 1:
            torch.distributed.barrier()

        self.epoch_init = load_checkpoint(
            cfg.training.ckpt_path,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )

        if self.dist.rank == 0:
            self.writer = SummaryWriter(log_dir=cfg.training.tensorboard_log_dir)

        # ── Sanity Check ──
        self._sanity_check(dataset)

    def _setup_validation(self, cfg, reader, logger0):
        self.num_validation_replicas = min(
            self.dist.world_size, cfg.training.num_validation_samples
        )
        self.num_validation_samples = (
            cfg.training.num_validation_samples
            // self.num_validation_replicas
            * self.num_validation_replicas
        )
        logger0.info(f"Number of validation samples: {self.num_validation_samples}")

        val_cfg = self.cfg.datapipe.copy()
        with open_dict(val_cfg):
            val_cfg.data_dir = self.cfg.training.raw_data_dir_validation
            val_cfg.num_samples = self.num_validation_samples

        val_dataset = instantiate(
            val_cfg,
            name="emag_validation",
            reader=reader,
            split="validation",
            logger=logger0,
            sample_type="all_time_steps",  # ← always full sequence for validation
        )

        if self.dist.rank < self.num_validation_replicas:
            val_sampler = None
            if self.dist.world_size > 1:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=self.num_validation_replicas,
                    rank=self.dist.rank,
                    shuffle=False,
                    drop_last=True,
                )
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                drop_last=True,
                pin_memory=True,
                num_workers=cfg.training.num_dataloader_workers,
                sampler=val_sampler,
                collate_fn=simsample_collate,
            )
        else:
            self.val_dataloader = DataLoader(
                torch.utils.data.Subset(val_dataset, []),
                batch_size=1,
            )

    def _sanity_check(self, dataset):
        """Verify data shapes and target layout."""
        if len(dataset) == 0:
            return

        sample = dataset[0]
        coords = sample.node_features["coords"]
        features = sample.node_features["features"]
        geometry = sample.node_features["geometry"]
        target = sample.node_target

        if self.dist.rank == 0:
            from physicsnemo.utils.logging import PythonLogger

            logger = PythonLogger("sanity")
            logger.info(f"\n{'─'*60}")
            logger.info("Sanity Check (Sample 0, normalized):")
            logger.info(f"  coords:   {list(coords.shape)}  range [{coords.min():.3f}, {coords.max():.3f}]")
            logger.info(f"  features: {list(features.shape)}  range [{features.min():.3f}, {features.max():.3f}]")
            logger.info(f"  geometry: {list(geometry.shape)}  range [{geometry.min():.3f}, {geometry.max():.3f}]")
            logger.info(f"  target:   {list(target.shape)}  range [{target.min():.3f}, {target.max():.3f}]")
            logger.info(f"  functional_dim: {coords.shape[-1] + features.shape[-1] + 1}  (coords + features + time)")
            logger.info(f"  out_dim: 1  (Fo = scalar |E|)")
            logger.info(f"  Training mode: TIME-CONDITIONAL (random timestep per iteration)")
            logger.info(f"{'─'*60}\n")

    def train(self, sample: SimSample):
        """Full train step: forward + backward."""
        self.optimizer.zero_grad()
        loss = self.forward(sample)
        self.backward(loss)
        return loss

    def forward(self, sample: SimSample):
        """
        Time-conditional forward.
        Time is already injected by datapipe. Target is already [N, 1].
        """
        with autocast(device_type="cuda", enabled=self.amp):
            pred = self.model(sample=sample, data_stats=self.data_stats)  # [N, 1]
            target = sample.node_target  # [N, 1]
            loss = self.criterion(pred, target)
        return loss

    def backward(self, loss):
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """
        Validation using full rollout (model.eval → _rollout path).
        Returns dict with MSE and per-timestep MSE.
        """
        self.model.eval()

        MSE = torch.zeros(1, device=self.dist.device)
        MSE_w_time = torch.zeros(self.rollout_steps, device=self.dist.device)

        for sample in self.val_dataloader:
            sample = sample[0].to(self.dist.device)

            # In eval mode, model returns [N, T, 1] via _rollout
            pred = self.model(sample=sample, data_stats=self.data_stats)
            target = sample.node_target

            T = min(pred.shape[1], target.shape[1])

            sq_error = torch.square(pred[:, :T] - target[:, :T])
            MSE_w_time[:T] += torch.mean(sq_error, dim=(0, 2))
            MSE += torch.mean(sq_error)

        # Sum errors across all ranks
        if self.dist.world_size > 1:
            torch.distributed.all_reduce(MSE, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(MSE_w_time, op=torch.distributed.ReduceOp.SUM)

        val_stats = {
            "MSE": MSE / self.num_validation_samples,
            "MSE_w_time": MSE_w_time / self.num_validation_samples,
        }

        self.model.train()
        return val_stats


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("train")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger0.file_logging()

    # Log full config and paths
    logger0.info(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    logger0.info(f"Output directory: {cfg.training.tensorboard_log_dir}")
    logger0.info(f"Checkpoint directory: {cfg.training.ckpt_path}")

    trainer = Trainer(cfg, logger0)

    logger0.info("=" * 60)
    logger0.info("Time-Conditional E-Field Training")
    logger0.info("=" * 60)
    logger0.info(f"  Epochs:          {cfg.training.epochs}")
    logger0.info(f"  Future steps:    {trainer.rollout_steps} (T-1)")
    logger0.info(f"  Mode:            TIME-CONDITIONAL (random t per step)")
    logger0.info(f"  Output (train):  [N, 1]  (single timestep)")
    logger0.info(f"  Output (eval):   [N, T, 1]  (full rollout)")
    logger0.info(f"  AMP:             {cfg.training.amp}")
    logger0.info(f"  World size:      {dist.world_size}")
    logger0.info("=" * 60)
    logger0.info("Training started...")

    for epoch in range(trainer.epoch_init, cfg.training.epochs):
        if trainer.sampler is not None:
            trainer.sampler.set_epoch(epoch)

        total_loss = 0.0
        num_batches = 0
        start = time.time()
        batch_start = start
        epoch_len = len(trainer.dataloader)
        log_every = max(1, epoch_len // 10)  # Log ~10 times per epoch

        for batch_idx, sample in enumerate(trainer.dataloader):
            sample = sample[0].to(dist.device)
            loss = trainer.train(sample)
            total_loss += loss.detach().item()
            num_batches += 1

            # Per-batch progress
            if (batch_idx + 1) % log_every == 0 or batch_idx == 0:
                batch_duration = time.time() - batch_start
                mem_gb = (
                    torch.cuda.memory_reserved() / 1024**3
                    if torch.cuda.is_available()
                    else 0.0
                )
                logger0.info(
                    f"Epoch {epoch + 1} [{batch_idx + 1}/{epoch_len}] "
                    f"Loss: {loss.detach().item():.6f} "
                    f"Duration: {batch_duration:.2f}s Mem: {mem_gb:.2f}GB"
                )
            batch_start = time.time()

        trainer.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        epoch_duration = time.time() - start
        logger0.info(
            f"Epoch {epoch + 1}/{cfg.training.epochs} "
            f"avg_loss: {avg_loss:.6f} "
            f"lr: {trainer.optimizer.param_groups[0]['lr']:.3e} "
            f"duration: {epoch_duration:.2f}s"
        )

        if dist.rank == 0:
            trainer.writer.add_scalar("train/loss", avg_loss, epoch)
            trainer.writer.add_scalar(
                "train/lr", trainer.optimizer.param_groups[0]["lr"], epoch
            )

        if dist.world_size > 1:
            torch.distributed.barrier()

        if dist.rank == 0 and (epoch + 1) % cfg.training.save_chckpoint_freq == 0:
            save_checkpoint(
                cfg.training.ckpt_path,
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch + 1,
            )
            logger0.info(f"Saved model on rank {dist.rank}")

        # Validation
        if (
            cfg.training.num_validation_samples > 0
            and (epoch + 1) % cfg.training.validation_freq == 0
        ):
            val_stats = trainer.validate(epoch)

            mse_val = val_stats["MSE"].item()
            mse_w_time = val_stats["MSE_w_time"]
            logger0.info(f"Validation epoch {epoch + 1}: MSE: {mse_val:.6f}")

            if _tabulate.available and dist.rank == 0:
                rows = [["MSE (overall)", f"{mse_val:.6f}"]]
                for i, m in enumerate(mse_w_time):
                    rows.append([f"timestep_{i}_MSE", f"{m.item():.6f}"])
                logger0.info(
                    f"\nValidation metrics:\n"
                    f"{_tabulate.tabulate(rows, headers=['Metric', 'Value'], tablefmt='pretty')}\n"
                )

            if dist.rank == 0:
                trainer.writer.add_scalar("val/MSE", mse_val, epoch)
                trainer.writer.add_scalar("val/RMSE", mse_val**0.5, epoch)
                for i in range(len(mse_w_time)):
                    trainer.writer.add_scalar(
                        f"val/timestep_{i}_MSE",
                        mse_w_time[i].item(),
                        epoch,
                    )

    logger0.info("Training completed!")
    if dist.rank == 0:
        trainer.writer.close()


if __name__ == "__main__":
    main()
