# datapipe.py
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Data Pipeline for Transient E-Field Magnitude Prediction with GeoTransolver / GALE.

Loads VTU + STL data via ema3d_reader containing:
    - Static field coordinates:    [N, 3]       (from VTU)
    - Static geometry positions:   [M, 3]       (from STL — for GALE ball queries)
    - Time-varying E_Magnitude:    [T, N, 1]    (from VTU)
    - Global parameters:           dict         (e.g. wave azimuth, frequency)

Pre-processing pipeline:
    1. Log-transform:    |E| -> log(|E| + eps)
    2. Z-score:          (log|E| - mean) / std
    3. Global params:    angles -> [sin, cos],  scalars -> value / divisor

Output format (per sample):
    node_features["coords"]:    [k, 3]       normalized field positions
    node_features["features"]:  [k, 1]       normalized log|E| at t=0
    node_features["geometry"]:  [k_geo, 3]   normalized geometry positions
    node_features["time"]:      scalar       (one_time_step mode only)
    node_target:                [k, 1] or [k, T-1, 1]
    global_features:            dict[str, Tensor]  encoded wave/material params
"""

import os
import json
import numpy as np
import torch
from typing import Callable, Optional, Any

from physicsnemo.utils.logging import PythonLogger

from ema3d_reader import Reader

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

STATS_DIRNAME = "stats"
NODE_STATS_FILE = "node_stats.json"
FEATURE_STATS_FILE = "feature_stats.json"
GEOMETRY_STATS_FILE = "geometry_stats.json"
EPS = 1e-8
LOG_EPS = 1e-1  # floor before log to avoid log(0)


# ═══════════════════════════════════════════════════════════════════════════════
# Point Cloud Sampling
# ═══════════════════════════════════════════════════════════════════════════════

def poisson_sample_indices_fixed(N: int, k: int, device=None) -> torch.Tensor:
    """Nearly-uniform random index sampler for large point clouds."""
    if k >= N:
        return torch.arange(N, device=device)

    gaps = torch.rand(k, device=device).exponential_()
    summed = gaps.sum()
    gaps *= N / summed
    idx = torch.cumsum(gaps, dim=0)
    idx -= gaps[0] / 2
    idx = torch.clamp(idx.floor().long(), min=0, max=N - 1)
    return idx


# ═══════════════════════════════════════════════════════════════════════════════
# JSON Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def save_json(data: dict, filepath: str) -> None:
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# Type Conversion Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _to_python_native(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    elif isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    elif isinstance(value, dict):
        return {k: _to_python_native(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_to_python_native(v) for v in value]
    elif hasattr(value, 'item'):
        return value.item()
    else:
        return value


def _to_tensor(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    elif isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy()).to(dtype=dtype)
    elif isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=dtype)
    else:
        return torch.tensor(value, dtype=dtype)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        return value
    else:
        return np.asarray(value)


def _stats_to_serializable(stats: dict) -> dict:
    return _to_python_native(stats)


def _stats_from_serializable(stats: dict, dtype: torch.dtype = torch.float32) -> dict:
    return {k: _to_tensor(v, dtype=dtype) for k, v in stats.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# SimSample Class
# ═══════════════════════════════════════════════════════════════════════════════

class SimSample:
    """
    Point cloud sample for GeoTransolver with GALE geometry context.

    Attributes:
        node_features: Dictionary containing:
            - "coords":    [N, 3] field grid positions (normalized)
            - "features":  [N, 1] log|E| at t=0 (normalized)
            - "geometry":  [M, 3] STL positions (normalized)
            - "time":      scalar (one_time_step mode)
        node_target: [N, 1] or [N, T-1, 1]
        global_features: Optional dict of encoded global parameters,
                         each tensor has shape [C_k].
    """

    def __init__(
        self,
        node_features: dict[str, torch.Tensor],
        node_target: torch.Tensor,
        global_features: Optional[dict[str, torch.Tensor]] = None,
    ):
        self.node_features = node_features
        self.node_target = node_target
        self.global_features = global_features

    def to(self, device: torch.device) -> "SimSample":
        self.node_features = {
            k: v.to(device) for k, v in self.node_features.items()
        }
        self.node_target = self.node_target.to(device)
        if self.global_features is not None:
            self.global_features = {
                k: v.to(device) for k, v in self.global_features.items()
            }
        return self

    def is_graph(self) -> bool:
        return False

    def get_info(self) -> dict:
        info = {
            "num_field_nodes": self.node_features["coords"].shape[0],
            "coords_shape": tuple(self.node_features["coords"].shape),
            "features_shape": tuple(self.node_features["features"].shape),
            "target_shape": tuple(self.node_target.shape),
        }
        if "geometry" in self.node_features:
            info["num_geometry_nodes"] = self.node_features["geometry"].shape[0]
            info["geometry_shape"] = tuple(self.node_features["geometry"].shape)
        if "time" in self.node_features:
            info["time"] = self.node_features["time"].item()
        if self.global_features is not None:
            info["global_features"] = {
                k: tuple(v.shape) for k, v in self.global_features.items()
            }
        return info

    def __repr__(self) -> str:
        N = self.node_features["coords"].shape[0]
        F = self.node_features["features"].shape[-1]
        T_target = self.node_target.shape[1] if self.node_target.ndim > 1 else 0
        M = self.node_features.get("geometry", torch.empty(0)).shape[0]
        G = len(self.global_features) if self.global_features else 0
        return f"SimSample(N={N}, M={M}, features={F}, T_target={T_target}, globals={G})"


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Class
# ═══════════════════════════════════════════════════════════════════════════════

class Dataset:
    NUM_FEATURES = 1

    def __init__(
        self,
        name: str = "dataset",
        reader: Optional[Callable] = None,
        data_dir: Optional[str] = None,
        split: str = "train",
        num_samples: int = 1000,
        num_steps: int = 99,
        logger=None,
        dt: float = 5e-3,
        debug: bool = False,
        log_transform: bool = True,
        stats_dir: Optional[str] = None,
        sample_type: str = "all_time_steps",
        resolution: Optional[int] = None,
        geometry_resolution: Optional[int] = None,
        global_params_keys: Optional[list] = None,
        global_params_normalization: Optional[dict] = None,
        **kwargs
    ):
        self.name = name
        self.data_dir = data_dir or "."
        self.split = split
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.logger = logger or PythonLogger()
        self.dt = dt
        self.debug = debug
        self.log_transform = log_transform
        self.sample_type = sample_type
        self.resolution = resolution
        self.geometry_resolution = geometry_resolution

        # Global params configuration
        self.global_params_keys = list(global_params_keys) if global_params_keys else []
        self.global_params_normalization = dict(global_params_normalization) if global_params_normalization else {}

        if sample_type not in ["all_time_steps", "one_time_step"]:
            raise ValueError(
                f"Invalid sample_type: {sample_type}. "
                f"Expected 'all_time_steps' or 'one_time_step'"
            )

        rollout_steps = num_steps - 1
        if sample_type == "one_time_step":
            self._resolve_idx = lambda idx: (idx // rollout_steps, idx % rollout_steps)
        else:
            self._resolve_idx = lambda idx: (idx, None)
        self._rollout_steps = rollout_steps

        self._log(f"\n{'='*70}")
        self._log(f"Initializing {self.__class__.__name__}")
        self._log(f"{'='*70}")
        self._log(f"  Name:        {name}")
        self._log(f"  Split:       {split}")
        self._log(f"  Data dir:    {self.data_dir}")
        self._log(f"  Num samples: {num_samples}")
        self._log(f"  Num steps:   {self.num_steps} (T)")
        self._log(f"  Rollout:     {self._rollout_steps} steps (T-1)")
        self._log(f"  Feature:     E_Magnitude (scalar |E|)")
        self._log(f"  Log-space:   {self.log_transform}")
        self._log(f"  sample_type: {self.sample_type}")
        self._log(f"  resolution:  {self.resolution}  (None = use all field points)")
        self._log(f"  geo_reso:    {self.geometry_resolution}  (None = use all STL points)")
        self._log(f"  global_keys: {self.global_params_keys}")

        # ── Stats directory ──
        if stats_dir is not None:
            self._stats_dir = stats_dir
        else:
            try:
                import hydra as _hydra
                base = _hydra.utils.get_original_cwd()
            except Exception:
                base = os.getcwd()
            self._stats_dir = os.path.join(base, STATS_DIRNAME)
        os.makedirs(self._stats_dir, exist_ok=True)
        self._log(f"  Stats dir:   {os.path.abspath(self._stats_dir)}")

        # ── Reader ──
        if reader is None:
            reader = Reader(debug=debug)

        point_data = reader(
            data_dir=self.data_dir,
            num_samples=num_samples,
            split=split,
            logger=self.logger,
        )

        if not point_data:
            raise ValueError(f"No data loaded from {self.data_dir}")

        # ── Storage ──
        self.mesh_pos_seq: list[torch.Tensor] = []
        self.e_magnitude_seq: list[torch.Tensor] = []
        self.geometry_seq: list[torch.Tensor] = []
        self.geometry_normals_seq: list[torch.Tensor] = []
        self.global_params_seq: list[dict] = []  # Raw global params per sample

        self._log(f"\n  Processing {len(point_data)} records...")

        for i, rec in enumerate(point_data):
            self._process_record(i, rec)

        self._log(f"  Loaded {len(self.mesh_pos_seq)} samples successfully")

        # ── Statistics & Normalization ──
        self._setup_statistics()
        self._apply_normalization()

        # ── Length / index logic ──
        self.length = len(self.mesh_pos_seq)
        if sample_type == "one_time_step":
            self._max_idx = self.length * self._rollout_steps
        else:
            self._max_idx = self.length

        self._log(f"  sample_type:   {self.sample_type}")
        self._log(f"  __len__:       {self._max_idx} "
                   f"({'samples × T' if sample_type == 'one_time_step' else 'samples'})")

        self._print_summary()

    def _log(self, msg: str):
        if self.debug:
            print(msg)
        if self.logger:
            self.logger.info(msg)

    def _process_record(self, idx: int, rec: dict):
        coords = _to_numpy(rec["field_coords"])
        N_nodes = coords.shape[0]

        if "geometry_pos" not in rec:
            raise ValueError(f"Record {idx} missing 'geometry_pos' field")
        geometry_pos = _to_numpy(rec["geometry_pos"])
        M_geo = geometry_pos.shape[0]

        geometry_normals = _to_numpy(rec.get("geometry_normals", np.zeros((M_geo, 3))))

        if "E_Magnitude" not in rec:
            raise ValueError(f"Record {idx} missing 'E_Magnitude' field")
        e_magnitude = _to_numpy(rec["E_Magnitude"])

        T_file = e_magnitude.shape[0]
        T = min(T_file, self.num_steps)

        if self.debug:
            print(f"\n    Record {idx}: N={N_nodes}, M={M_geo}, T={T} (available: {T_file})")

        coords_seq = np.tile(coords[np.newaxis, :, :], (T, 1, 1))
        self.mesh_pos_seq.append(torch.from_numpy(coords_seq.copy()).float())

        e_mag_sliced = e_magnitude[:T].copy()

        if self.log_transform:
            raw_min = e_mag_sliced.min()
            raw_max = e_mag_sliced.max()
            e_mag_sliced = np.log(np.maximum(e_mag_sliced, 0.0) + LOG_EPS)
            if self.debug:
                print(f"      Raw |E| range:    [{raw_min:.4e}, {raw_max:.4e}]")
                print(f"      Log |E| range:    [{e_mag_sliced.min():.4e}, {e_mag_sliced.max():.4e}]")
        else:
            if self.debug:
                print(f"      |E| range:        [{e_mag_sliced.min():.4e}, {e_mag_sliced.max():.4e}]")

        self.e_magnitude_seq.append(torch.from_numpy(e_mag_sliced).float())
        self.geometry_seq.append(torch.from_numpy(geometry_pos.copy()).float())
        self.geometry_normals_seq.append(torch.from_numpy(geometry_normals.copy()).float())

        # ── Extract global parameters ──
        raw_globals = rec.get("global_params", {})
        if self.global_params_keys:
            missing = [k for k in self.global_params_keys if k not in raw_globals]
            if missing:
                raise KeyError(
                    f"Record {idx}: missing global params {missing}. "
                    f"Available keys in record: {list(raw_globals.keys())}. "
                    f"Configured keys: {self.global_params_keys}"
                )
            # Keep only configured keys, convert to float
            filtered = {k: float(raw_globals[k]) for k in self.global_params_keys}
        else:
            filtered = {}

        self.global_params_seq.append(filtered)

        if self.debug:
            print(f"      coords:       {coords_seq.shape}")
            print(f"      geometry:     {geometry_pos.shape}")
            label = "log(|E|+eps)" if self.log_transform else "|E|"
            print(f"      {label}:      {e_mag_sliced.shape}")
            if filtered:
                print(f"      global_params: {filtered}")

    def _setup_statistics(self):
        node_stats_path = os.path.join(self._stats_dir, NODE_STATS_FILE)
        feat_stats_path = os.path.join(self._stats_dir, FEATURE_STATS_FILE)
        geo_stats_path = os.path.join(self._stats_dir, GEOMETRY_STATS_FILE)

        if self.split == "train":
            self._log("\n  Computing statistics from training data...")
            self.node_stats = self._compute_node_stats()
            self.feature_stats = self._compute_feature_stats()
            self.geometry_stats = self._compute_geometry_stats()

            save_json(_stats_to_serializable(self.node_stats), node_stats_path)
            save_json(_stats_to_serializable(self.feature_stats), feat_stats_path)
            save_json(_stats_to_serializable(self.geometry_stats), geo_stats_path)
            self._log(f"  Saved statistics to {self._stats_dir}/")
        else:
            if all(os.path.exists(p) for p in [node_stats_path, feat_stats_path, geo_stats_path]):
                self._log(f"\n  Loading statistics from {self._stats_dir}/")
                self.node_stats = _stats_from_serializable(load_json(node_stats_path))
                self.feature_stats = _stats_from_serializable(load_json(feat_stats_path))
                self.geometry_stats = _stats_from_serializable(load_json(geo_stats_path))
            else:
                self._log("\n  WARNING: No saved statistics found, computing from current split")
                self.node_stats = self._compute_node_stats()
                self.feature_stats = self._compute_feature_stats()
                self.geometry_stats = self._compute_geometry_stats()

        self._log_statistics()

    def _log_statistics(self):
        pos_mean = self.node_stats['pos_mean']
        pos_std = self.node_stats['pos_std']
        feat_mean = self.feature_stats['feature_mean']
        feat_std = self.feature_stats['feature_std']
        geo_mean = self.geometry_stats['geo_pos_mean']
        geo_std = self.geometry_stats['geo_pos_std']

        space_label = "log-space" if self.log_transform else "linear-space"

        self._log(f"\n  Statistics ({space_label}):")
        if isinstance(pos_mean, torch.Tensor):
            self._log(f"    pos_mean:      [{pos_mean[0].item():.6f}, {pos_mean[1].item():.6f}, {pos_mean[2].item():.6f}]")
            self._log(f"    pos_std:       [{pos_std[0].item():.6f}, {pos_std[1].item():.6f}, {pos_std[2].item():.6f}]")
        if isinstance(feat_mean, torch.Tensor):
            self._log(f"    feature_mean:  {feat_mean.item():.6e}  ({space_label})")
            self._log(f"    feature_std:   {feat_std.item():.6e}  ({space_label})")
            if feat_std.item() > EPS:
                self._log(f"    std/mean ratio: {abs(feat_std.item() / (feat_mean.item() + EPS)):.3f}")
        if isinstance(geo_mean, torch.Tensor):
            self._log(f"    geo_pos_mean:  [{geo_mean[0].item():.6f}, {geo_mean[1].item():.6f}, {geo_mean[2].item():.6f}]")
            self._log(f"    geo_pos_std:   [{geo_std[0].item():.6f}, {geo_std[1].item():.6f}, {geo_std[2].item():.6f}]")

    def _compute_node_stats(self) -> dict:
        all_pos = torch.cat([p.reshape(-1, 3) for p in self.mesh_pos_seq], dim=0)
        mean = torch.mean(all_pos, dim=0)
        std = torch.std(all_pos, dim=0)
        std = torch.clamp(std, min=EPS)
        return {"pos_mean": mean, "pos_std": std}

    def _compute_feature_stats(self) -> dict:
        all_feat = torch.cat([f.reshape(-1, 1) for f in self.e_magnitude_seq], dim=0)
        mean = torch.mean(all_feat, dim=0)
        std = torch.std(all_feat, dim=0)
        std = torch.clamp(std, min=EPS)

        if self.debug:
            space = "log" if self.log_transform else "linear"
            self._log(f"    Feature stats ({space}): mean={mean.item():.4f}, std={std.item():.4f}")
            self._log(f"    Feature range ({space}): [{all_feat.min().item():.4f}, {all_feat.max().item():.4f}]")

        return {"feature_mean": mean, "feature_std": std}

    def _compute_geometry_stats(self) -> dict:
        all_geo = torch.cat(self.geometry_seq, dim=0)
        mean = torch.mean(all_geo, dim=0)
        std = torch.std(all_geo, dim=0)
        std = torch.clamp(std, min=EPS)
        return {"geo_pos_mean": mean, "geo_pos_std": std}

    def _apply_normalization(self):
        self._log("\n  Applying normalization...")

        pos_mean = _to_tensor(self.node_stats["pos_mean"])
        pos_std = _to_tensor(self.node_stats["pos_std"])
        feat_mean = _to_tensor(self.feature_stats["feature_mean"])
        feat_std = _to_tensor(self.feature_stats["feature_std"])
        geo_mean = _to_tensor(self.geometry_stats["geo_pos_mean"])
        geo_std = _to_tensor(self.geometry_stats["geo_pos_std"])

        for i in range(len(self.mesh_pos_seq)):
            self.mesh_pos_seq[i] = (
                (self.mesh_pos_seq[i] - pos_mean.view(1, 1, -1))
                / pos_std.view(1, 1, -1)
            )
            self.e_magnitude_seq[i] = (
                (self.e_magnitude_seq[i] - feat_mean.view(1, 1, -1))
                / feat_std.view(1, 1, -1)
            )
            self.geometry_seq[i] = (
                (self.geometry_seq[i] - geo_mean.view(1, -1))
                / geo_std.view(1, -1)
            )

    def _build_global_features(self, raw_params: dict) -> Optional[dict]:
        """
        Encode raw global parameters into normalized tensors.

        Normalization types (from global_params_normalization config):
            - "angle_deg":  [sin(θ), cos(θ)]  -> 2 features
            - "angle_rad":  [sin(θ), cos(θ)]  -> 2 features
            - "scale":      value / divisor    -> 1 feature
            - "log_scale":  log(value/divisor + eps)  -> 1 feature

        Returns dict of tensors (shape [C_k]) or None if no params configured.
        """
        if not self.global_params_keys or not raw_params:
            return None

        features = {}
        for key in self.global_params_keys:
            value = raw_params[key]
            cfg = self.global_params_normalization.get(
                key, {"type": "scale", "divisor": 1.0}
            )
            norm_type = cfg.get("type", "scale")

            if norm_type == "angle_deg":
                rad = float(value) * np.pi / 180.0
                features[key] = torch.tensor(
                    [np.sin(rad), np.cos(rad)], dtype=torch.float32
                )
            elif norm_type == "angle_rad":
                rad = float(value)
                features[key] = torch.tensor(
                    [np.sin(rad), np.cos(rad)], dtype=torch.float32
                )
            elif norm_type == "scale":
                divisor = float(cfg.get("divisor", 1.0))
                features[key] = torch.tensor(
                    [float(value) / divisor], dtype=torch.float32
                )
            elif norm_type == "log_scale":
                divisor = float(cfg.get("divisor", 1.0))
                features[key] = torch.tensor(
                    [float(np.log(float(value) / divisor + 1e-8))],
                    dtype=torch.float32,
                )
            else:
                raise ValueError(
                    f"Unknown normalization type '{norm_type}' for key '{key}'. "
                    f"Supported: angle_deg, angle_rad, scale, log_scale"
                )

        return features

    def get_global_dim(self) -> int:
        """
        Compute the total encoded dimension of global features.

        Useful for configuring the model's global_dim parameter.
        """
        total = 0
        for key in self.global_params_keys:
            cfg = self.global_params_normalization.get(
                key, {"type": "scale", "divisor": 1.0}
            )
            norm_type = cfg.get("type", "scale")
            if norm_type in ("angle_deg", "angle_rad"):
                total += 2
            elif norm_type in ("scale", "log_scale"):
                total += 1
            else:
                raise ValueError(f"Unknown norm type: {norm_type}")
        return total

    def _print_summary(self):
        space_label = "log(|E|+eps)" if self.log_transform else "|E|"

        self._log(f"\n{'='*70}")
        self._log(f"Dataset Summary: {self.name} ({self.split})")
        self._log(f"{'='*70}")
        self._log(f"  Total samples:     {self.length}")
        self._log(f"  Time steps (T):    {self.num_steps}")
        self._log(f"  Target steps:      {self.num_steps - 1} (T-1)")
        self._log(f"  Feature:           {space_label}")
        self._log(f"  Feature dimension: {self.NUM_FEATURES}")
        self._log(f"  Log-transform:     {self.log_transform}")

        if self.length > 0:
            sample = self[0]
            N_sampled = sample.node_features['coords'].shape[0]
            M_sampled = sample.node_features['geometry'].shape[0]
            N_full = self.mesh_pos_seq[0].shape[1]
            M_full = self.geometry_seq[0].shape[0]

            self._log(f"\n  Sample 0 shapes (normalized{', log-space' if self.log_transform else ''}):")
            self._log(f"    coords:    {list(sample.node_features['coords'].shape)}   "
                       f"[k={N_sampled} of N={N_full}]")
            self._log(f"    features:  {list(sample.node_features['features'].shape)}  [k={N_sampled}, 1]")
            self._log(f"    geometry:  {list(sample.node_features['geometry'].shape)}  "
                       f"[k_geo={M_sampled} of M={M_full}]")
            self._log(f"    target:    {list(sample.node_target.shape)}")

            # Per-epoch sampling info
            if self.resolution is not None and self.resolution < N_full:
                reduction_pct = 100 * (1 - N_sampled / N_full)
                self._log(f"\n  Per-epoch point sampling active:")
                self._log(f"    Field:    {N_full:,} → {N_sampled:,}  ({reduction_pct:.1f}% reduction)")
            if self.geometry_resolution is not None and self.geometry_resolution < M_full:
                geo_reduction = 100 * (1 - M_sampled / M_full)
                self._log(f"    Geometry: {M_full:,} → {M_sampled:,}  ({geo_reduction:.1f}% reduction)")

            # Global features info
            if sample.global_features is not None:
                self._log(f"\n  Global features (sample 0):")
                for k, v in sample.global_features.items():
                    raw_val = self.global_params_seq[0].get(k)
                    self._log(f"    {k}: raw={raw_val}, encoded={v.tolist()}")
                global_dim = self.get_global_dim()
                self._log(f"    Total global_dim: {global_dim}")

            # Report normalized value ranges
            feat = sample.node_features['features']
            tgt = sample.node_target
            self._log(f"\n  Normalized ranges (sample 0):")
            self._log(f"    features:  [{feat.min().item():.3f}, {feat.max().item():.3f}]")
            self._log(f"    target:    [{tgt.min().item():.3f}, {tgt.max().item():.3f}]")

            self._log(f"\n  GeoTransolver config:")
            self._log(f"    geometry_dim   = 3")
            self._log(f"    functional_dim = 5  (coords + {space_label} + time)")
            self._log(f"    out_dim        = 1")
            if self.global_params_keys:
                self._log(f"    global_dim     = {self.get_global_dim()}")

        self._log(f"{'='*70}\n")

    def __len__(self) -> int:
        return self._max_idx

    def __getitem__(self, idx: int) -> SimSample:
        """
        Fetch a training/validation sample.

        If `resolution` is set, randomly subsamples field points per call.
        If `geometry_resolution` is set, randomly subsamples STL points per call.
        Same field indices are used across all T timesteps for temporal consistency.

        Returns:
            SimSample with:
              - node_features["coords"]:   [k, 3]
              - node_features["features"]: [k, 1]
              - node_features["geometry"]: [k_geo, 3]
              - node_features["time"]:     scalar (one_time_step mode)
              - node_target:               [k, 1] or [k, T-1, 1]
              - global_features:           dict of encoded params (or None)
        """
        batch_idx, time_idx = self._resolve_idx(idx)

        pos_seq = self.mesh_pos_seq[batch_idx]
        e_mag_seq = self.e_magnitude_seq[batch_idx]
        geometry = self.geometry_seq[batch_idx]
        raw_globals = self.global_params_seq[batch_idx]

        N = pos_seq.shape[1]
        M = geometry.shape[0]

        # ── Random subsample field points ──
        if self.resolution is not None and self.resolution < N:
            field_idx = poisson_sample_indices_fixed(
                N, self.resolution, device=pos_seq.device
            )
            pos_seq_sampled = pos_seq[:, field_idx, :]
            e_mag_seq_sampled = e_mag_seq[:, field_idx, :]
        else:
            pos_seq_sampled = pos_seq
            e_mag_seq_sampled = e_mag_seq

        # ── Random subsample geometry points ──
        if self.geometry_resolution is not None and self.geometry_resolution < M:
            geo_idx = poisson_sample_indices_fixed(
                M, self.geometry_resolution, device=geometry.device
            )
            geometry_sampled = geometry[geo_idx]
        else:
            geometry_sampled = geometry

        # ── Build encoded global features ──
        global_features = self._build_global_features(raw_globals)

        # ── Build SimSample ──
        node_features = {
            "coords": pos_seq_sampled[0],
            "features": e_mag_seq_sampled[0],
            "geometry": geometry_sampled,
        }

        T = e_mag_seq_sampled.shape[0]

        if time_idx is not None:
            # one_time_step mode
            node_features["time"] = torch.tensor(
                time_idx / self._rollout_steps, dtype=torch.float32
            )
            node_target = e_mag_seq_sampled[time_idx + 1]

            return SimSample(
                node_features=node_features,
                node_target=node_target,
                global_features=global_features,
            )
        else:
            # all_time_steps mode
            if T > 1:
                node_target = e_mag_seq_sampled[1:].permute(1, 0, 2)
            else:
                k = pos_seq_sampled.shape[1]
                node_target = torch.zeros((k, 0, 1), dtype=torch.float32)

            return SimSample(
                node_features=node_features,
                node_target=node_target,
                global_features=global_features,
            )


def simsample_collate(batch: list[SimSample]) -> list[SimSample]:
    return batch
