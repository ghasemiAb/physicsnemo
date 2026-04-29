# inference.py
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Inference Script for Time-Conditional GeoTransolver E-Field Magnitude Prediction.

Loads trained model and runs inference on test data, saving predicted
|E| values to VTU files with preserved mesh structure for visualization.

Data flow:
    Input:  field_coords [N, 3], |E| at t=0 [N, 1], geometry [M, 3],
            (optional) global_params dict (wave angle, frequency, ...)
    Model:  _rollout loops T steps, each conditioned on t_norm -> [N, T, 1]
    Output: VTU files with predicted / ground truth / error |E| per timestep
            + PVD animation file for ParaView

Pre-processing pipeline (must match training):
    1. Log-transform:   |E| -> log(max(|E|, 0) + LOG_EPS)
    2. Z-score:         (log|E| - mean) / std
    3. Denormalize:     un-zscore -> exp() -> clamp(0) -> physical |E|

GeoTransolver mapping (per timestep):
    local_embedding  = coords + log|E|_0 + t_norm  [B, N, functional_dim=5]
    local_positions  = field_coords                 [B, N, 3]
    geometry         = STL positions                [B, M, 3]  (ball_query for GALE)
    global_embedding = encoded wave params           [B, 1, G]  (optional)
    output           = [B, N, Fo=1]                 scalar log|E| at that timestep

Global parameters:
    If `global_params_keys` is configured in the datapipe config, each test case
    MUST have a global_params.json file at:
        <case>_Animation/global_params.json
    containing the required keys (e.g., wave_azimuth, wave_elevation, frequency_ghz).
    If global params are disabled (empty keys list), no JSON is required.

Stats directory:
    Auto-resolved to <launch_dir>/stats/ using hydra.utils.get_original_cwd().
    No config entry needed — just launch training and inference from the same directory.
"""

import os
import sys
import time
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(__file__))

import hydra
from hydra.utils import to_absolute_path, instantiate
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils import load_checkpoint

from datapipe import simsample_collate

EPS = 1e-8
LOG_EPS = 1e-1  # must match datapipe.py


# ═══════════════════════════════════════════════════════════════════════════════
# Logging Utilities
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TimestepStats:
    """Statistics for a single timestep of scalar |E| field."""
    mean: float
    std: float
    min_val: float
    max_val: float
    active_pct: float

    @classmethod
    def from_array(cls, arr: np.ndarray, threshold: float = 1e-3) -> "TimestepStats":
        arr = arr.flatten()
        return cls(
            mean=float(arr.mean()),
            std=float(arr.std()),
            min_val=float(arr.min()),
            max_val=float(arr.max()),
            active_pct=float((arr > threshold).sum() / len(arr) * 100),
        )


def print_header(title: str, width: int = 80):
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def print_subheader(title: str, width: int = 80):
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def print_config_table(config: dict, title: str = "Configuration"):
    print(f"\n  ┌{'─' * 60}┐")
    print(f"  │  {title:<56}  │")
    print(f"  ├{'─' * 60}┤")
    for key, value in config.items():
        print(f"  │  {key:<25}: {str(value):<30}  │")
    print(f"  └{'─' * 60}┘")


def print_prediction_stats_header():
    print("\n  ┌────────┬──────────────────────────────────────┬──────────────────────────────────────┬────────────────────┐")
    print("  │        │            PREDICTION                │            GROUND TRUTH              │       ERROR        │")
    print("  │ Step   ├──────────┬──────────┬────────┬───────┼──────────┬──────────┬────────┬───────┼──────────┬─────────┤")
    print("  │        │  |E|Mean │  |E|Std  │ |E|Max │Active%│  |E|Mean │  |E|Std  │ |E|Max │Active%│   MAE    │  RMSE   │")
    print("  ├────────┼──────────┼──────────┼────────┼───────┼──────────┼──────────┼────────┼───────┼──────────┼─────────┤")


def print_prediction_stats_row(
    timestep: int,
    pred_stats: TimestepStats,
    gt_stats: TimestepStats = None,
    mae: float = None,
    rmse: float = None,
):
    if gt_stats is not None:
        print(
            f"  │ t={timestep:3d}  │ {pred_stats.mean:8.2e} │ {pred_stats.std:8.2e} │{pred_stats.max_val:7.1e} │{pred_stats.active_pct:5.1f}% │"
            f" {gt_stats.mean:8.2e} │ {gt_stats.std:8.2e} │{gt_stats.max_val:7.1e} │{gt_stats.active_pct:5.1f}% │"
            f" {mae:8.2e} │ {rmse:7.2e} │"
        )
    else:
        print(
            f"  │ t={timestep:3d}  │ {pred_stats.mean:8.2e} │ {pred_stats.std:8.2e} │{pred_stats.max_val:7.1e} │{pred_stats.active_pct:5.1f}% │"
            f"     -    │     -    │   -    │  -    │"
            f"     -    │    -    │"
        )


def print_prediction_stats_footer():
    print("  └────────┴──────────┴──────────┴────────┴───────┴──────────┴──────────┴────────┴───────┴──────────┴─────────┘")


def print_summary_stats(
    total_mae: float,
    total_rmse: float,
    total_mse: float,
    num_timesteps: int,
    has_ground_truth: bool,
):
    print("\n  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │                    OVERALL STATISTICS                           │")
    print("  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │  Total timesteps predicted:  {num_timesteps:<32} │")
    if has_ground_truth:
        print(f"  │  Mean Absolute Error (MAE):  {total_mae:<32.6e} │")
        print(f"  │  Root Mean Square Error:     {total_rmse:<32.6e} │")
        print(f"  │  Mean Square Error (MSE):    {total_mse:<32.6e} │")
    else:
        print(f"  │  Ground truth:               {'Not available':<32} │")
    print("  └─────────────────────────────────────────────────────────────────┘")


def print_run_summary(
    run_name: str,
    num_field_pts: int,
    num_geo_pts: int,
    num_timesteps: int,
    output_dir: str,
):
    print("\n  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │                       CASE SUMMARY                              │")
    print("  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │  Case name:          {run_name:<42}│")
    print(f"  │  Field points (N):   {num_field_pts:<42,}│")
    print(f"  │  Geometry pts (M):   {num_geo_pts:<42,}│")
    print(f"  │  Timesteps:          {num_timesteps:<42}│")
    print(f"  │  Output directory:   {output_dir:<42}│")
    print("  └─────────────────────────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════════════════════════
# Tensor Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _to_tensor(value, dtype=torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _stats_to_device(stats: dict, device: torch.device, dtype=torch.float32) -> dict:
    return {k: _to_tensor(v, dtype=dtype).to(device) for k, v in stats.items()}


def denormalize_emag(
    y: torch.Tensor,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    log_transform: bool = True,
) -> torch.Tensor:
    """
    Denormalize |E| predictions.

    Pipeline (log_transform=True):
        normalized log-space  →  un-normalize  →  log-space  →  exp()  →  |E|
    """
    if y.ndim == 2:
        out = y * feat_std.view(1, -1) + feat_mean.view(1, -1)
    elif y.ndim == 3:
        out = y * feat_std.view(1, 1, -1) + feat_mean.view(1, 1, -1)
    else:
        raise AssertionError(f"Expected [N,1], [N,T,1], or [T,N,1], got {y.shape}")

    if log_transform:
        out = torch.exp(out) - LOG_EPS
        out = torch.clamp(out, min=0.0)

    return out


def denormalize_coords(
    coords: torch.Tensor, pos_mean: torch.Tensor, pos_std: torch.Tensor
) -> torch.Tensor:
    """Denormalize coordinates [N, 3]."""
    return coords * pos_std.view(1, -1) + pos_mean.view(1, -1)


def denormalize_geometry(
    geometry: torch.Tensor,
    geo_pos_mean: torch.Tensor,
    geo_pos_std: torch.Tensor,
) -> torch.Tensor:
    """Denormalize geometry positions [M, 3]."""
    return geometry * geo_pos_std.view(1, -1) + geo_pos_mean.view(1, -1)


# ═══════════════════════════════════════════════════════════════════════════════
# VTU Saving
# ═══════════════════════════════════════════════════════════════════════════════

def save_vtu_predictions(
    coords: torch.Tensor,
    preds: list[torch.Tensor],
    output_dir: str,
    vtu_template_path: str | None = None,
    prefix: str = "frame",
    compute_error: bool = True,
    verbose: bool = True,
    gt_seq: list[torch.Tensor] | None = None,
    geometry: torch.Tensor | None = None,
) -> dict:
    """Save predicted |E| values to VTU files with preserved mesh topology."""
    os.makedirs(output_dir, exist_ok=True)

    coords_np = coords.detach().cpu().numpy()
    N = coords_np.shape[0]
    T = len(preds)

    all_pred_stats = []
    all_gt_stats = []
    all_mae = []
    all_rmse = []
    gt_available_count = 0

    if geometry is not None:
        geo_np = geometry.detach().cpu().numpy()
        geo_mesh = pv.PolyData(geo_np)
        geo_file = os.path.join(output_dir, "geometry.vtp")
        geo_mesh.save(geo_file)
        if verbose:
            print(f"\n  Saved geometry ({geo_np.shape[0]} pts) to {geo_file}")

    if vtu_template_path and os.path.exists(vtu_template_path):
        template_mesh = pv.read(vtu_template_path)
    else:
        template_mesh = pv.PolyData(coords_np).cast_to_unstructured_grid()
        logging.warning("No valid VTU template found. Falling back to casted point cloud.")

    if verbose:
        print_prediction_stats_header()

    for t in range(T):
        timestep = t + 1
        pred_np = preds[t].detach().cpu().numpy().squeeze(-1)

        if pred_np.shape[0] != N:
            logging.warning(f"Point mismatch at t={timestep}")
            continue

        pred_stats = TimestepStats.from_array(pred_np)
        all_pred_stats.append(pred_stats)

        gt_np = None
        gt_stats = None
        mae = None
        rmse = None

        if compute_error and gt_seq is not None and t < len(gt_seq):
            try:
                gt_np = gt_seq[t].detach().cpu().numpy().squeeze()
                gt_stats = TimestepStats.from_array(gt_np)
                all_gt_stats.append(gt_stats)

                error = pred_np - gt_np
                mae = float(np.abs(error).mean())
                rmse = float(np.sqrt((error ** 2).mean()))

                all_mae.append(mae)
                all_rmse.append(rmse)
                gt_available_count += 1
            except Exception as e:
                logging.warning(f"Error computing GT stats at t={timestep}: {e}")

        if verbose:
            print_prediction_stats_row(timestep, pred_stats, gt_stats, mae, rmse)

        mesh = template_mesh.copy()
        mesh.point_data["E_Magnitude_pred"] = pred_np

        if gt_np is not None and gt_np.shape[0] == N:
            mesh.point_data["E_Magnitude_exact"] = gt_np
            mesh.point_data["E_Magnitude_error"] = pred_np - gt_np
            mesh.point_data["E_Magnitude_abs_error"] = np.abs(pred_np - gt_np)

        out_file = os.path.join(output_dir, f"{prefix}_{timestep:03d}_pred.vtu")
        mesh.save(out_file)

    if verbose:
        print_prediction_stats_footer()

    stats = {
        "num_timesteps": T,
        "gt_available_count": gt_available_count,
        "has_ground_truth": gt_available_count > 0,
    }

    if gt_available_count > 0:
        stats["total_mae"] = float(np.mean(all_mae))
        stats["total_rmse"] = float(np.mean(all_rmse))
        stats["total_mse"] = float(np.mean([r ** 2 for r in all_rmse]))

    _write_pvd(output_dir, prefix, T)

    return stats


def _write_pvd(output_dir: str, prefix: str, num_timesteps: int):
    """Write a PVD collection file for ParaView animation."""
    pvd_content = ['<?xml version="1.0"?>']
    pvd_content.append('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">')
    pvd_content.append('  <Collection>')
    for t in range(1, num_timesteps + 1):
        filename = f"{prefix}_{t:03d}_pred.vtu"
        pvd_content.append(f'    <DataSet timestep="{t}" group="" part="0" file="{filename}"/>')
    pvd_content.append('  </Collection>')
    pvd_content.append('</VTKFile>')

    pvd_path = os.path.join(output_dir, "prediction_animation.pvd")
    with open(pvd_path, 'w') as f:
        f.write('\n'.join(pvd_content))


# ═══════════════════════════════════════════════════════════════════════════════
# Stats & Global Params Verification
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_stats_dir() -> str:
    """Resolve the stats directory to <launch_dir>/stats/."""
    try:
        base = hydra.utils.get_original_cwd()
    except Exception:
        base = os.getcwd()
    return os.path.join(base, "stats")


def verify_stats_dir(stats_dir: str) -> dict:
    """Verify that training stats exist and return their contents."""
    required = {
        "node": os.path.join(stats_dir, "node_stats.json"),
        "feature": os.path.join(stats_dir, "feature_stats.json"),
        "geometry": os.path.join(stats_dir, "geometry_stats.json"),
    }

    missing = {k: v for k, v in required.items() if not os.path.exists(v)}
    if missing:
        try:
            launch_dir = hydra.utils.get_original_cwd()
        except Exception:
            launch_dir = os.getcwd()

        raise FileNotFoundError(
            f"Training stats not found!\n\n"
            f"  Missing files:\n"
            + "\n".join(f"    {v}" for v in missing.values())
            + f"\n\n"
            f"  Looked in:   {stats_dir}\n"
            f"  Launch dir:  {launch_dir}\n"
            f"  Hydra cwd:   {os.getcwd()}\n\n"
            f"  Fix: Run training first from the same directory,\n"
            f"  or copy stats/ to {launch_dir}/stats/"
        )

    loaded = {}
    for key, path in required.items():
        with open(path, "r") as f:
            loaded[key] = json.load(f)

    return loaded


def check_global_params_ready(case_path: str, required_keys: list, logger) -> dict:
    """
    Verify global_params.json exists and contains all required keys for a test case.

    If `required_keys` is empty (global params disabled), this is a no-op and
    returns an empty dict.

    Args:
        case_path: Path to the *_Animation directory.
        required_keys: List of keys that must be present (from config).
        logger: Logger instance.

    Returns:
        Loaded global_params dict.

    Raises:
        FileNotFoundError: If JSON is missing but required.
        KeyError: If JSON is present but missing required keys.
    """
    if not required_keys:
        return {}

    gp_path = os.path.join(case_path, "global_params.json")
    if not os.path.exists(gp_path):
        raise FileNotFoundError(
            f"Global parameters are configured but global_params.json is missing:\n"
            f"  Case:      {case_path}\n"
            f"  Expected:  {gp_path}\n"
            f"  Required:  {required_keys}\n\n"
            f"  Fix: create {gp_path} with:\n"
            f'         {{ "{required_keys[0]}": <value>, ... }}\n'
            f"  OR:    set global_params_keys: [] in datapipe config to disable globals"
        )

    with open(gp_path, "r") as f:
        params = json.load(f)

    missing = [k for k in required_keys if k not in params]
    if missing:
        raise KeyError(
            f"global_params.json is missing required keys:\n"
            f"  File:      {gp_path}\n"
            f"  Missing:   {missing}\n"
            f"  Found:     {list(params.keys())}\n"
            f"  Required:  {required_keys}"
        )

    return params


# ═══════════════════════════════════════════════════════════════════════════════
# Inference Worker
# ═══════════════════════════════════════════════════════════════════════════════

class InferenceWorker:
    """
    Inference worker for time-conditional GeoTransolver |E| prediction.

    In eval mode the model's _rollout() method iterates over T timesteps,
    conditioning each forward pass on t_norm = t / T, and returns [N, T, 1].

    Global parameters (wave angle, frequency, etc.) are automatically carried
    through the pipeline via sample.global_features if configured in the
    datapipe. The model's _build_global_embedding() assembles them and passes
    to GeoTransolver's global_embedding input.

    Stats are auto-resolved to <launch_dir>/stats/ — no config needed.
    Both training and inference must be launched from the same directory.
    """

    def __init__(self, cfg: DictConfig, logger: PythonLogger, dist: DistributedManager):
        self.cfg = cfg
        self.logger = logger
        self.dist = dist
        self.device = dist.device

        if dist.rank == 0:
            print_header("TIME-CONDITIONAL GEOTRANSOLVER |E| MAGNITUDE INFERENCE")

        # ── Build and load model ──
        self.model = instantiate(cfg.model)
        logging.getLogger().setLevel(logging.INFO)
        self.model.to(self.device)
        self.model.eval()

        ckpt_path = cfg.training.ckpt_path
        load_checkpoint(ckpt_path, models=self.model, device=self.device)

        # ── Configuration ──
        self.vtu_prefix = cfg.inference.get("vtu_prefix", "frame")
        self.write_vtu = cfg.inference.get("write_vtu", True)
        self.compute_error = cfg.inference.get("compute_error", True)
        self.output_dir = cfg.inference.get("output_dir", "./predictions")
        self.verbose = cfg.inference.get("verbose", True)

        self.num_future_steps = cfg.training.num_time_steps - 1
        self.num_workers = cfg.training.num_dataloader_workers

        # ── Global params config ──
        self.global_keys = list(cfg.datapipe.get("global_params_keys", []) or [])
        self.global_dim = cfg.model.get("global_dim", None)

        # ── Resolve and verify training stats ──
        self.stats_dir = resolve_stats_dir()
        loaded_stats = verify_stats_dir(self.stats_dir)

        # ── Log configuration ──
        if dist.rank == 0:
            print_config_table({
                "Checkpoint": ckpt_path,
                "Output directory": self.output_dir,
                "Stats directory": self.stats_dir,
                "VTU prefix": self.vtu_prefix,
                "Prediction mode": "Time-conditional (T forward passes)",
                "Future steps (T-1)": self.num_future_steps,
                "Per-step input dim": "functional_dim=5 (coords+log|E|_0+t)",
                "Per-step output dim": "Fo=1 (scalar log|E|)",
                "Final output layout": "[N, T, 1] (from _rollout)",
                "Log-transform": True,
                "LOG_EPS": LOG_EPS,
                "Global params": ", ".join(self.global_keys) if self.global_keys else "disabled",
                "Global dim": self.global_dim if self.global_keys else "—",
                "Compute error": self.compute_error,
                "Device": str(self.device),
            }, title="Inference Configuration")

            feat = loaded_stats["feature"]
            print(f"\n  ✓ Training stats verification:")
            print(f"    feature_mean (log-space): {feat['feature_mean']}")
            print(f"    feature_std  (log-space): {feat['feature_std']}")
            print(f"    Loaded from: {self.stats_dir}")

            if self.global_keys:
                print(f"\n  ⚠ Global parameters are ENABLED — each test case must have:")
                print(f"    <case>_Animation/global_params.json")
                print(f"    with keys: {self.global_keys}")
            else:
                print(f"\n  Global parameters are DISABLED — JSON files not required.")

        self.logger.info(f"[Rank {dist.rank}] Loaded checkpoint {ckpt_path}")
        self.logger.info(f"[Rank {dist.rank}] Using training stats from {self.stats_dir}")

    @torch.no_grad()
    def run_on_single_case(self, case_path: str):
        """Process a single simulation case (*_Animation directory)."""
        case_name = os.path.basename(case_path)

        if self.verbose:
            print_subheader(f"Processing Case: {case_name}")

        self.logger.info(f"[Rank {self.dist.rank}] Processing case: {case_name}")

        # ── Verify global_params.json exists (if globals are configured) ──
        case_globals = check_global_params_ready(
            case_path, self.global_keys, self.logger
        )

        if self.verbose:
            if self.global_keys:
                print(f"\n  Global parameters for this case:")
                for k, v in case_globals.items():
                    print(f"    {k}: {v}")
            else:
                print(f"\n  Global parameters: disabled (no keys configured)")

        with tempfile.TemporaryDirectory() as tmpdir:
            symlink_path = os.path.join(tmpdir, case_name)
            os.symlink(case_path, symlink_path)

            # ── Instantiate reader explicitly ──
            reader = instantiate(self.cfg.reader)

            # ── Build dataset with TRAINING stats ──
            # global_params_keys / global_params_normalization come from cfg.datapipe
            dataset = instantiate(
                self.cfg.datapipe,
                name="emag_inference",
                reader=reader,
                split="test",
                num_steps=self.cfg.training.num_time_steps,
                num_samples=1,
                logger=self.logger,
                data_dir=tmpdir,
                stats_dir=self.stats_dir,
                sample_type="all_time_steps",  # FORCE full sequence for inference
                resolution=None,                # use ALL field points
                geometry_resolution=None,        # use ALL geometry points
            )

            data_stats = dict(
                node=_stats_to_device(dataset.node_stats, self.device),
                feature=_stats_to_device(dataset.feature_stats, self.device),
                geometry=_stats_to_device(dataset.geometry_stats, self.device),
            )

            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=self.num_workers,
                collate_fn=simsample_collate,
            )

            pos_mean = data_stats["node"]["pos_mean"]
            pos_std = data_stats["node"]["pos_std"]
            feat_mean = data_stats["feature"]["feature_mean"]
            feat_std = data_stats["feature"]["feature_std"]
            geo_pos_mean = data_stats["geometry"]["geo_pos_mean"]
            geo_pos_std = data_stats["geometry"]["geo_pos_std"]

            if self.verbose:
                print(f"\n  Normalization Statistics (from training):")
                print(f"    Stats dir:      {self.stats_dir}")
                print(f"    Position mean:  [{pos_mean[0].item():.6f}, {pos_mean[1].item():.6f}, {pos_mean[2].item():.6f}]")
                print(f"    Position std:   [{pos_std[0].item():.6f}, {pos_std[1].item():.6f}, {pos_std[2].item():.6f}]")
                print(f"    |E| mean (log): {feat_mean.item():.6e}")
                print(f"    |E| std  (log): {feat_std.item():.6e}")
                print(f"    Geo pos mean:   [{geo_pos_mean[0].item():.6f}, {geo_pos_mean[1].item():.6f}, {geo_pos_mean[2].item():.6f}]")
                print(f"    Geo pos std:    [{geo_pos_std[0].item():.6f}, {geo_pos_std[1].item():.6f}, {geo_pos_std[2].item():.6f}]")

            for local_idx, sample in enumerate(dataloader):
                if isinstance(sample, list):
                    sample = sample[0]
                sample = sample.to(self.device)

                # ── Log global features present on the sample ──
                if self.verbose and sample.global_features is not None:
                    print(f"\n  Global features (encoded, normalized):")
                    for k, v in sample.global_features.items():
                        raw = case_globals.get(k, "n/a")
                        print(f"    {k}: raw={raw}, encoded={v.cpu().numpy().tolist()}  "
                              f"(shape {list(v.shape)})")

                # ── Input statistics ──
                input_emag = sample.node_features["features"]
                input_emag_denorm = denormalize_emag(
                    input_emag, feat_mean, feat_std, log_transform=True
                )
                input_emag_np = input_emag_denorm.cpu().numpy().flatten()

                N_field = input_emag.shape[0]
                M_geo = sample.node_features["geometry"].shape[0]

                if self.verbose:
                    print(f"\n  Input (t=0) Statistics:")
                    print(f"    Field points (N):  {N_field:,}")
                    print(f"    Geometry pts (M):  {M_geo:,}")
                    print(f"    |E| mean:          {input_emag_np.mean():.6e}")
                    print(f"    |E| std:           {input_emag_np.std():.6e}")
                    print(f"    |E| range:         [{input_emag_np.min():.4e}, {input_emag_np.max():.4e}]")
                    print(f"    Active (>1e-3):    {(input_emag_np > 1e-3).sum() / len(input_emag_np) * 100:.1f}%")

                    raw_norm = input_emag.cpu().numpy().flatten()
                    print(f"    Normalized range:  [{raw_norm.min():.4f}, {raw_norm.max():.4f}]")

                # ── Time-conditional rollout ──
                if self.verbose:
                    print(f"\n  Running time-conditional rollout ({self.num_future_steps} timesteps)...")
                    print(f"    Each step: [N, 5] -> model -> [N, 1], conditioned on t_norm")
                    if self.global_keys:
                        print(f"    Global context: {len(self.global_keys)} params "
                              f"-> [1, 1, {self.global_dim}] embedding (fixed across time)")

                t_start = time.time()

                pred = self.model(sample=sample, data_stats=data_stats)

                t_elapsed = time.time() - t_start

                if self.verbose:
                    print(f"  Rollout completed in {t_elapsed:.3f}s "
                          f"({t_elapsed / self.num_future_steps * 1000:.1f} ms/step)")
                    print(f"  Output shape: {list(pred.shape)} "
                          f"(expected [N={N_field}, T={self.num_future_steps}, 1])")

                    raw_pred = pred.cpu().numpy()
                    print(f"  Raw model output range (norm log): "
                          f"[{raw_pred.min():.4f}, {raw_pred.max():.4f}]")

                # ── Convert to time-first [T, N, 1] ──
                pred_time_first = pred.permute(1, 0, 2)

                # ── Denormalize coordinates, geometry ──
                coords_norm = sample.node_features["coords"]
                coords = denormalize_coords(coords_norm, pos_mean, pos_std)

                geometry_norm = sample.node_features["geometry"]
                geometry = denormalize_geometry(geometry_norm, geo_pos_mean, geo_pos_std)

                # ── Denormalize predictions ──
                T_pred = pred_time_first.shape[0]
                pred_seq_denorm = [
                    denormalize_emag(
                        pred_time_first[t], feat_mean, feat_std, log_transform=True
                    )
                    for t in range(T_pred)
                ]

                # ── Denormalize ground truth ──
                gt_target = sample.node_target
                gt_time_first = gt_target.permute(1, 0, 2)
                gt_seq_denorm = [
                    denormalize_emag(
                        gt_time_first[t], feat_mean, feat_std, log_transform=True
                    )
                    for t in range(gt_time_first.shape[0])
                ]

                # ── Sanity check ──
                if self.verbose and len(pred_seq_denorm) > 0 and len(gt_seq_denorm) > 0:
                    p0 = pred_seq_denorm[0].cpu().numpy().flatten()
                    g0 = gt_seq_denorm[0].cpu().numpy().flatten()
                    print(f"\n  Denormalized sanity (t=1):")
                    print(f"    Pred |E|:  mean={p0.mean():.4e}, "
                          f"range=[{p0.min():.4e}, {p0.max():.4e}]")
                    print(f"    GT   |E|:  mean={g0.mean():.4e}, "
                          f"range=[{g0.min():.4e}, {g0.max():.4e}]")
                    gt_mean = g0.mean()
                    if gt_mean > 1e-10:
                        print(f"    Ratio (pred_mean / gt_mean): "
                              f"{p0.mean() / gt_mean:.4f}")

                # ── Save VTU files ──
                if self.write_vtu:
                    out_dir = os.path.join(
                        self.output_dir, f"rank{self.dist.rank}", case_name
                    )

                    vtu_files = sorted(list(Path(case_path).rglob("*.vtu")))
                    template_path = str(vtu_files[0]) if vtu_files else None
                    if not template_path:
                        self.logger.warning(f"No VTU template found in {case_path}")

                    # Save global params alongside predictions for provenance
                    if case_globals:
                        os.makedirs(out_dir, exist_ok=True)
                        with open(os.path.join(out_dir, "global_params_used.json"), "w") as f:
                            json.dump(case_globals, f, indent=2)

                    stats = save_vtu_predictions(
                        coords=coords,
                        preds=pred_seq_denorm,
                        output_dir=out_dir,
                        vtu_template_path=template_path,
                        prefix=self.vtu_prefix,
                        compute_error=self.compute_error,
                        verbose=self.verbose,
                        gt_seq=gt_seq_denorm,
                        geometry=geometry,
                    )

                    if self.verbose:
                        print_summary_stats(
                            total_mae=stats.get("total_mae", 0),
                            total_rmse=stats.get("total_rmse", 0),
                            total_mse=stats.get("total_mse", 0),
                            num_timesteps=stats["num_timesteps"],
                            has_ground_truth=stats["has_ground_truth"],
                        )

                        print_run_summary(
                            run_name=case_name,
                            num_field_pts=N_field,
                            num_geo_pts=M_geo,
                            num_timesteps=T_pred,
                            output_dir=out_dir,
                        )

                    self.logger.info(f"[Rank {self.dist.rank}] Saved to {out_dir}")

            self.logger.info(f"[Rank {self.dist.rank}] Finished case: {case_name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("inference")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger0.file_logging()
    logging.getLogger().setLevel(logging.INFO)

    parent_dir = to_absolute_path(cfg.inference.raw_data_dir_test)

    if not os.path.isdir(parent_dir):
        logger0.error(f"Parent directory not found: {parent_dir}")
        return

    case_dirs = sorted([
        d.path for d in os.scandir(parent_dir)
        if d.is_dir() and d.name.endswith("_Animation")
    ])

    if len(case_dirs) == 0:
        logger0.error(f"No *_Animation directories found under: {parent_dir}")
        return

    if dist.rank == 0:
        print_header("DATA DISCOVERY")
        print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
        print(f"  │  Data directory: {parent_dir:<45} │")
        print(f"  │  Found {len(case_dirs)} case(s):{' ' * 50}│")
        for i, case_dir in enumerate(case_dirs[:10]):
            case_name = os.path.basename(case_dir)
            print(f"  │    {i + 1:2d}. {case_name:<55} │")
        if len(case_dirs) > 10:
            print(f"  │    ... and {len(case_dirs) - 10} more{' ' * 47}│")
        print(f"  └─────────────────────────────────────────────────────────────────┘")

    logger0.info(f"Found {len(case_dirs)} cases under {parent_dir}")

    my_cases = case_dirs[dist.rank :: dist.world_size]

    if dist.rank == 0:
        print(f"\n  Distribution across {dist.world_size} rank(s):")
        print(f"    Rank {dist.rank}: {len(my_cases)} case(s) assigned")

    logger.info(f"[Rank {dist.rank}] Assigned {len(my_cases)} cases.")

    worker = InferenceWorker(cfg, logger, dist)

    for i, case_path in enumerate(my_cases):
        if dist.rank == 0:
            print(f"\n  Progress: {i + 1}/{len(my_cases)} cases")
        worker.run_on_single_case(case_path)

    if dist.rank == 0:
        print_header("INFERENCE COMPLETE")
        print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
        print(f"  │  ✓ Successfully processed {len(my_cases)} case(s){' ' * 34}│")
        print(f"  │  ✓ Output saved to: {worker.output_dir:<41} │")
        print(f"  │  ✓ Mode: Time-conditional [N,T,1] ({worker.num_future_steps} steps){' ' * 12}│")
        print(f"  │  ✓ Stats from: {worker.stats_dir:<45} │")
        if worker.global_keys:
            print(f"  │  ✓ Global context: {len(worker.global_keys)} params, dim={worker.global_dim}{' ' * 30}│")
        print(f"  └─────────────────────────────────────────────────────────────────┘\n")

    logger0.info("Inference completed successfully.")


if __name__ == "__main__":
    main()
