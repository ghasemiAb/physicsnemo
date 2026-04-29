# rollout.py
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Time-Conditional GeoTransolver E-Field Magnitude Prediction with Fourier Features.

Each forward pass predicts |E| at a single future timestep, conditioned on
a normalised time embedding appended to the input features.

Optional Fourier features lift spatial coordinates into a higher-dimensional
space to mitigate the neural network's spectral bias — critical for problems
with sharp spatial structure like aperture-coupled EM field penetration.

Architecture:
    - local_embedding:   [1, N, functional_dim]
                         = FourierEncode(coords) + |E|_0 + t_norm
    - local_positions:   [1, N, 3]  (RAW coords — required for GALE ball queries)
    - geometry:          [1, M, 3] or [1, M, geometry_enc_dim]
    - global_embedding:  [1, 1, G]  (encoded wave/global params, optional)
    - output:            [1, N, Fo]  where Fo = 1 (scalar |E|)

Fourier Features:
    Supports two encoding strategies:

    1. "dyadic" — Classic positional encoding with geometric progression
       of frequencies: 2^0 π, 2^1 π, 2^2 π, ..., 2^(L-1) π
       Deterministic, well-understood, used in NeRF and Transformers.

    2. "gaussian" — Random Fourier Features sampled from N(0, σ² I)
       Often better for regression tasks on smooth manifolds. Introduces
       more spatial frequency diversity at the cost of a random seed
       dependency.

Training:  random timestep sampled via datapipe → predict [N, 1] → MSE loss
Inference: loop over T steps → stack to [N, T, 1]

References:
    - Tancik et al., "Fourier Features Let Networks Learn High Frequency
      Functions in Low Dimensional Domains", NeurIPS 2020.
    - Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance
      Fields for View Synthesis", ECCV 2020.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Literal

from physicsnemo.experimental.models.geotransolver import GeoTransolver
from datapipe import SimSample

EPS = 1e-8


# ═══════════════════════════════════════════════════════════════════════════════
# Fourier Features
# ═══════════════════════════════════════════════════════════════════════════════

class FourierFeatures(nn.Module):
    """
    Positional encoding via Fourier features.

    Two encoding modes:

    1. 'dyadic' — Deterministic geometric progression:
        freqs = [2^0 π, 2^1 π, ..., 2^(L-1) π]

        Output per input dim:
            [..., x, sin(2^0 π x), cos(2^0 π x), sin(2^1 π x), cos(2^1 π x), ...]

        For input_dim=D, num_freqs=L, include_input=True:
            output_dim = D + 2 * D * L

    2. 'gaussian' — Random Fourier Features (RFF):
        Sample B ~ N(0, σ²) of shape [L, D].
        For input x, compute projections: xB^T  of shape [N, L]

        Output:
            [..., x, sin(2π xB^T), cos(2π xB^T)]

        For input_dim=D, num_freqs=L, include_input=True:
            output_dim = D + 2 * L

    Notes:
        * `include_input=True` is strongly recommended — the raw coordinate
          values carry information that the MLP can use directly.
        * The frequency matrix is registered as a buffer (not a parameter),
          so it moves with .to(device) and is saved in checkpoints but NOT
          updated by the optimizer.
        * For gaussian mode, the random B matrix is generated once at __init__
          using a fixed seed for reproducibility.

    Args:
        input_dim:     Dimensionality of input coordinates (typically 3 for x,y,z)
        num_freqs:     Number of frequency bands (L)
        mode:          'dyadic' or 'gaussian'
        include_input: Concatenate original input to output (recommended)
        sigma:         Scale for Gaussian mode only (typical 1.0 to 50.0)
        seed:          RNG seed for Gaussian mode reproducibility
    """

    def __init__(
        self,
        input_dim: int = 3,
        num_freqs: int = 6,
        mode: Literal["dyadic", "gaussian"] = "dyadic",
        include_input: bool = True,
        sigma: float = 10.0,
        seed: int = 42,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.mode = mode
        self.include_input = include_input
        self.sigma = sigma

        if mode == "dyadic":
            # freqs: [L] = [2^0 π, 2^1 π, ..., 2^(L-1) π]
            freqs = 2.0 ** torch.arange(num_freqs).float() * math.pi
            self.register_buffer("freqs", freqs)
            self.register_buffer(
                "B", torch.empty(0)
            )  # placeholder for consistent state_dict

        elif mode == "gaussian":
            # B: [L, D] sampled from N(0, sigma^2)
            # Use a generator with fixed seed for reproducibility
            g = torch.Generator().manual_seed(seed)
            B = torch.randn(num_freqs, input_dim, generator=g) * sigma
            self.register_buffer("B", B)
            self.register_buffer(
                "freqs", torch.empty(0)
            )  # placeholder for consistent state_dict

        else:
            raise ValueError(
                f"Unknown Fourier mode: {mode!r}. Expected 'dyadic' or 'gaussian'."
            )

    @property
    def output_dim(self) -> int:
        """Compute output dimensionality."""
        base = self.input_dim if self.include_input else 0

        if self.mode == "dyadic":
            # sin + cos for each (dim, freq) pair
            return base + 2 * self.input_dim * self.num_freqs
        else:  # gaussian
            # sin + cos for each projected frequency
            return base + 2 * self.num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., input_dim] coordinate tensor

        Returns:
            Encoded tensor of shape [..., output_dim]
        """
        if self.mode == "dyadic":
            # x: [..., D] -> [..., D, 1]
            # freqs: [L]
            # Broadcast product: [..., D, L]
            xf = x.unsqueeze(-1) * self.freqs.view(
                *([1] * (x.ndim - 1)), 1, -1
            )
            # Flatten the last two dims: [..., D*L]
            xf = xf.flatten(-2, -1)
            sin_feat = torch.sin(xf)
            cos_feat = torch.cos(xf)

        else:  # gaussian
            # x: [..., D], B: [L, D]
            # x @ B.T: [..., L]
            xf = 2.0 * math.pi * (x @ self.B.T)
            sin_feat = torch.sin(xf)
            cos_feat = torch.cos(xf)

        if self.include_input:
            return torch.cat([x, sin_feat, cos_feat], dim=-1)
        else:
            return torch.cat([sin_feat, cos_feat], dim=-1)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, num_freqs={self.num_freqs}, "
            f"mode={self.mode!r}, include_input={self.include_input}, "
            f"output_dim={self.output_dim}, sigma={self.sigma}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Time-Conditional GeoTransolver with Fourier Features
# ═══════════════════════════════════════════════════════════════════════════════

class GeoTransolverTimeConditionalEField(GeoTransolver):
    """
    GeoTransolver with time-conditional training for E-field magnitude,
    optionally enhanced with Fourier positional encoding of coordinates.

    Time conditioning:
        Each forward pass receives a scalar t_norm ∈ [0, 1] appended to the
        per-point feature vector. At inference, _rollout() iterates over all
        T-1 future timesteps and stacks results to [N, T-1, 1].

    Fourier features:
        If `use_fourier=True`, the input coordinates are passed through a
        Fourier encoder that expands [x, y, z] to [x, y, z, sin(...), cos(...)].
        This mitigates spectral bias and improves accuracy on problems with
        sharp spatial structure (e.g., EM field penetration through apertures).

        IMPORTANT: Only the `local_embedding` channel receives Fourier features.
        The `local_positions` input to GALE ball queries MUST remain in raw
        3D coordinate space for the ball query to work correctly.

    Global parameters:
        If the sample carries `global_features` (e.g., wave azimuth, frequency),
        they are assembled into a [1, 1, G] embedding and passed to
        GeoTransolver's `global_embedding` input.

    Args (extends GeoTransolver):
        dt:                  Physical time step (stored but not used in loss)
        num_time_steps:      Total T including initial state
        use_fourier:         Enable Fourier encoding of coordinates
        fourier_num_freqs:   Number of frequency bands (L)
        fourier_mode:        'dyadic' or 'gaussian'
        fourier_sigma:       Gaussian scale (only used when mode='gaussian')
        fourier_seed:        RNG seed for gaussian mode (reproducibility)
        fourier_on_geometry: Also apply Fourier encoding to geometry input
    """

    Fo = 1  # Output features per timestep (scalar |E|)

    def __init__(self, *args, **kwargs):
        # ── Time conditioning parameters ──
        self.dt: float = kwargs.pop("dt", 5e-3)
        num_time_steps: int = kwargs.pop("num_time_steps")
        self.rollout_steps = num_time_steps - 1

        # ── Fourier feature parameters ──
        self.use_fourier: bool = kwargs.pop("use_fourier", False)
        fourier_num_freqs: int = kwargs.pop("fourier_num_freqs", 6)
        fourier_mode: str = kwargs.pop("fourier_mode", "dyadic")
        fourier_sigma: float = kwargs.pop("fourier_sigma", 10.0)
        fourier_seed: int = kwargs.pop("fourier_seed", 42)
        fourier_on_geometry: bool = kwargs.pop("fourier_on_geometry", False)
        fourier_include_input: bool = kwargs.pop("fourier_include_input", True)

        # ── Build Fourier encoders BEFORE super().__init__ so we can
        #    compute the effective functional_dim and geometry_dim ──
        if self.use_fourier:
            self.fourier_coords = FourierFeatures(
                input_dim=3,
                num_freqs=fourier_num_freqs,
                mode=fourier_mode,
                include_input=fourier_include_input,
                sigma=fourier_sigma,
                seed=fourier_seed,
            )
            coords_encoded_dim = self.fourier_coords.output_dim

            # Recompute functional_dim:
            # Original = 3 (coords) + 1 (|E|_0) + 1 (time) = 5
            # New      = coords_encoded_dim + 1 + 1
            orig_functional_dim = kwargs.get("functional_dim", 5)
            # The non-coord portion (|E|_0 + time) has size (orig - 3)
            non_coord_dim = orig_functional_dim - 3
            new_functional_dim = coords_encoded_dim + non_coord_dim
            kwargs["functional_dim"] = new_functional_dim
        else:
            self.fourier_coords = None

        if self.use_fourier and fourier_on_geometry:
            self.fourier_geometry = FourierFeatures(
                input_dim=3,
                num_freqs=fourier_num_freqs,
                mode=fourier_mode,
                include_input=fourier_include_input,
                sigma=fourier_sigma,
                seed=fourier_seed + 1,  # different seed from coords
            )
            kwargs["geometry_dim"] = self.fourier_geometry.output_dim
        else:
            self.fourier_geometry = None

        # ── Validate out_dim ──
        out_dim: int = kwargs.get("out_dim")
        if out_dim is not None and out_dim < self.Fo:
            raise ValueError(
                f"out_dim={out_dim} must be >= Fo={self.Fo} for "
                f"time-conditional prediction."
            )

        # Pop keys that parent GeoTransolver doesn't accept
        kwargs.pop("initial_vel", None)

        # ── Build the base GeoTransolver with adjusted dims ──
        super().__init__(*args, **kwargs)

        # ── Log the configuration ──
        self._print_config()

    def _print_config(self):
        """Print Fourier configuration for debugging."""
        print(f"\n{'─' * 60}")
        print(f"GeoTransolverTimeConditionalEField Configuration:")
        print(f"{'─' * 60}")
        print(f"  num_time_steps:       {self.rollout_steps + 1}")
        print(f"  rollout_steps:        {self.rollout_steps}")
        print(f"  dt:                   {self.dt}")
        print(f"  Fo (output per step): {self.Fo}")
        print(f"  use_fourier:          {self.use_fourier}")

        if self.use_fourier:
            print(f"  fourier_coords:       {self.fourier_coords}")
            if self.fourier_geometry is not None:
                print(f"  fourier_geometry:     {self.fourier_geometry}")
            else:
                print(f"  fourier_on_geometry:  False")
        print(f"{'─' * 60}\n")

    def _build_global_embedding(
        self, sample: SimSample
    ) -> Optional[torch.Tensor]:
        """
        Assemble a [1, 1, G] global embedding tensor from sample.global_features.

        Returns:
            Tensor of shape [1, 1, G] or None if no global features.
        """
        if sample.global_features is None or len(sample.global_features) == 0:
            return None

        ordered = [sample.global_features[k] for k in sample.global_features]
        g = torch.cat(ordered, dim=-1)  # [G]
        return g.unsqueeze(0).unsqueeze(0)  # [1, 1, G]

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        """Dispatch to _forward (training) or _rollout (inference)."""
        if self.training:
            return self._forward(sample, data_stats)
        else:
            return self._rollout(sample, data_stats)

    def _forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        """
        Single-timestep forward pass (used during training).

        Returns:
            pred: [N, 1] predicted log|E| at the conditioned timestep
        """
        inputs = sample.node_features
        coords = inputs["coords"]           # [N, 3]
        features = inputs.get(
            "features",
            coords.new_zeros((coords.size(0), 0)),
        )                                    # [N, 1] or [N, 0]
        geometry = inputs["geometry"]        # [M, 3]
        time_val = inputs["time"]            # scalar

        N = coords.size(0)

        # ── Build time feature: broadcast scalar to [N, 1] ──
        t_feat = time_val.unsqueeze(0).expand(N, 1)

        # ── Fourier-encode coordinates for local_embedding ──
        # IMPORTANT: local_positions (passed to GALE ball_query) must use
        # RAW 3D coords — Fourier encoding only applies to local_embedding.
        if self.fourier_coords is not None:
            coords_for_embedding = self.fourier_coords(coords)  # [N, encoded_dim]
        else:
            coords_for_embedding = coords

        # ── Fourier-encode geometry (optional) ──
        if self.fourier_geometry is not None:
            geometry_encoded = self.fourier_geometry(geometry)
        else:
            geometry_encoded = geometry

        # ── Concatenate inputs ──
        # functional_dim = coords_encoded_dim + 1(|E|_0) + 1(time)
        fx_t = torch.cat(
            [coords_for_embedding, features, t_feat], dim=-1
        )  # [N, functional_dim]

        # ── Global embedding from wave parameters ──
        global_emb = self._build_global_embedding(sample)

        # ── Forward through GeoTransolver ──
        pred = (
            super(GeoTransolverTimeConditionalEField, self)
            .forward(
                local_embedding=fx_t.unsqueeze(0),           # [1, N, functional_dim]
                geometry=geometry_encoded.unsqueeze(0),       # [1, M, geometry_dim]
                local_positions=coords.unsqueeze(0),         # [1, N, 3]  ← RAW coords!
                global_embedding=global_emb,                  # [1, 1, G] or None
            )
            .squeeze(0)
        )  # [N, out_dim]

        return pred[:, : self.Fo]  # [N, 1]

    def _rollout(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        """
        Full rollout at inference time.

        Iterates over all T-1 future timesteps, setting time = t / T for each.
        Fourier encoding and global features remain fixed across the rollout.

        Returns:
            pred: [N, T-1, 1] predicted |E| at each future timestep
        """
        device = sample.node_features["coords"].device
        outputs: list[torch.Tensor] = []

        for t in range(self.rollout_steps):
            time_val = torch.tensor(
                t / self.rollout_steps, device=device, dtype=torch.float32
            )
            sample.node_features["time"] = time_val

            y_t = self._forward(sample, data_stats)  # [N, 1]
            outputs.append(y_t)

        return torch.stack(outputs, dim=0).transpose(0, 1)  # [N, T, 1]
