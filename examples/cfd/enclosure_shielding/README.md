# Transient Electromagnetic Field Prediction with Time-Conditional GeoTransolver

A neural surrogate for predicting the time evolution of electromagnetic
fields inside shielded enclosures with apertures, built on
**GeoTransolver** from NVIDIA PhysicsNeMo and trained on simulations
generated with **Ansys EMA3D**.

The model predicts the spatial distribution of the electric-field
magnitude $|E|$ at future timesteps, given:

- The initial field distribution at $t = 0$
- The 3D geometry of the enclosure (STL surface mesh)
- Optional global parameters (incident wave direction, frequency,
  polarization, ...)

---

## Problem Overview

When an electromagnetic wave impinges on a metallic enclosure with slot
apertures, the wave penetrates through the apertures and produces a
complex time-varying field distribution inside. Predicting this internal
field is critical for:

- **Shielding effectiveness analysis** of electronic enclosures
- **EMC/EMI compliance** in automotive, aerospace, and defense
  applications
- **Rapid design-space exploration** for aperture geometry optimization

Conventional FDTD simulations are accurate but computationally
expensive, requiring minutes to hours per configuration. This example
trains a neural surrogate that produces predictions in a few hundred
milliseconds, enabling real-time design iteration.

```text
                Incident EM Wave
                       │
                       ▼
      ┌────────────────────────────────┐
      │                                │
      │    ┌─────┐    ┌─────┐          │
   ───┤    │slot │    │slot │          │
 wave ┤    └─────┘    └─────┘          │
   ───┤        ↓ penetration ↓         │
      │                                │
      │      ┌─────────────────┐       │
      │      │  Internal Field │       │
      │      │  |E|(x, y, z, t)│       │
      │      └─────────────────┘       │
      │                                │
      └────────────────────────────────┘
```

---

## Model Overview

The model is built on **GeoTransolver**, a geometry-aware transformer
from PhysicsNeMo's experimental models. It uses Geometry-Aware Local
Equivariant (GALE) attention to incorporate the enclosure geometry
throughout the forward pass via cross-attention between the field
points and downsampled STL surface points.

Key architectural decisions for this EM application:

1. **Time-conditional prediction.** Each forward pass predicts the
   field at one timestep conditioned on a normalized time embedding
   $t/T$. At inference, the model is called $T-1$ times to produce the
   full temporal sequence. This approach is adapted from the
   PhysicsNeMo crash simulation recipe.

2. **Log-space preprocessing.** $|E|$ values span orders of magnitude
   and include many near-zero points. The preprocessing pipeline
   applies $\log(|E| + \epsilon)$ before normalization and $\exp(\cdot)$
   at denormalization, which compresses the dynamic range into a
   near-Gaussian distribution and **guarantees non-negative
   predictions**.

3. **Fourier feature encoding.** Coordinates are optionally lifted into
   a higher-dimensional frequency space to mitigate spectral bias. Both
   dyadic (NeRF-style) and Gaussian Random Fourier Features are
   supported.

4. **Per-epoch random point sampling.** The datapipe uses Poisson
   sampling to select a different random subset of field points per
   iteration, enabling training on large meshes that exceed GPU memory.

5. **Global parameters.** Wave direction angles, frequency, and
   amplitude are encoded and injected through GeoTransolver's
   `global_embedding` input, where they modulate every transformer
   block.

---

## Getting Started

### Prerequisites

This example requires:

- `physicsnemo` (with `physicsnemo.experimental.models.geotransolver`)
- `pyvista` for VTU / STL I/O
- `tensordict` (dependency of GeoTransolver's GALE kernels)
- `hydra-core`, `omegaconf` for configuration
- `tabulate`, `torchinfo` (optional, for training metrics)

### Installation

```bash
pip install physicsnemo pyvista tensordict hydra-core omegaconf tabulate torchinfo
```

> **Note:** A CUDA-capable GPU is strongly recommended for training. An
> NVIDIA A100 (40 GB) or equivalent is sufficient for the default
> configuration.

---

## Data Preparation

Each simulation case is stored in a directory whose name ends with the
`_Animation` suffix:

```text
data/train/
├── Enclosure_slot_124_Animation/
│   ├── stl/
│   │   └── Enclosure_slot_124_Geometry.stl     # Enclosure surface mesh
│   ├── vtu/
│   │   ├── frame_001.vtu                       # E-field at t = 0
│   │   ├── frame_002.vtu                       # E-field at t = 1
│   │   ├── ...
│   │   └── frame_N.vtu                         # E-field at t = N-1
│   └── global_params.json                      # Optional: wave parameters
├── Enclosure_slot_146_Animation/
└── ...
```

### Required VTU point data

| Array         | Shape     | Description                |
|---------------|-----------|----------------------------|
| `E_Magnitude` | `[N, 1]`  | Scalar $\|E\|$ at each grid point |

### Optional `global_params.json`

```json
{
  "wave_azimuth": 30.0,
  "wave_elevation": 45.0,
  "frequency_ghz": 1.5
}
```

### Train / val / test split

Organize the data into train, validation, and test splits. An example
layout for a dataset of 23 simulations:

```text
data/ready_data/
├── train/   (20 cases)
├── val/     (2 cases)
└── test/    (1 case)
```

---

## Data Analysis (Recommended)

Before training, analyze the $|E|$-field distribution to verify data
quality and choose an appropriate `LOG_EPS` value for the log-transform:

```bash
python analyze_efield_distribution.py /path/to/data/train \
    --output-dir ./efield_analysis
```

This produces:

- Percentile distribution tables (console + CSV)
- Zero / near-zero point analysis
- Log-space statistics for several candidate `LOG_EPS` values
- Histograms and CDF plots
- A data-driven recommendation for `LOG_EPS`

For most EM shielding datasets, `LOG_EPS = 0.1` produces the tightest
distribution (`std / |mean|` ratio near 2.0) while clamping only
physically insignificant near-zero values.

---

## Configuration

The example uses **Hydra** for configuration. The main entry point is
`conf/config.yaml`, which composes model, datapipe, training, and
inference sub-configs.

### Model configuration

`conf/model/geotransolver_time_conditional.yaml`:

```yaml
_target_: rollout.GeoTransolverTimeConditionalEField
_convert_: all

# Per-point features: coords(3) + log|E|_0(1) + time(1) = 5
# Auto-adjusted when Fourier features are enabled.
functional_dim: 5
out_dim: 1
geometry_dim: 3

# Global parameters encoding (set to null if no global_params.json)
global_dim: 5                      # 2 (azimuth) + 2 (elevation) + 1 (freq)

# Fourier feature encoding (optional, significantly helps EM problems)
use_fourier: true
fourier_num_freqs: 6
fourier_mode: "dyadic"             # "dyadic" or "gaussian"
fourier_include_input: true
fourier_on_geometry: false

# Transformer architecture
slice_num: 128
n_layers: 8
n_hidden: 256
n_head: 8
use_te: false
time_input: false
include_local_features: true

num_time_steps: ${training.num_time_steps}
dt: 5e-3
```

### Training configuration

`conf/training/default.yaml`:

```yaml
raw_data_dir: "/path/to/data/train"
raw_data_dir_validation: "/path/to/data/val"

epochs: 2000
num_time_steps: 20

# Optimizer
start_lr: 5e-5
min_lr: 1e-6
weight_decay: 1e-4

# Data
num_samples: 20
num_validation_samples: 2
validation_freq: 50
save_chckpoint_freq: 100

amp: true
num_dataloader_workers: 8
```

### Datapipe configuration

`conf/datapipe/default.yaml`:

```yaml
_target_: datapipe.Dataset
log_transform: true
sample_type: "one_time_step"

# Per-epoch random sampling for large meshes (optional)
resolution: 50000                  # Sample 50k field points per iteration
geometry_resolution: 20000         # Sample 20k STL points per iteration

# Global parameters (leave empty to disable)
global_params_keys:
  - wave_azimuth
  - wave_elevation
  - frequency_ghz

global_params_normalization:
  wave_azimuth:
    type: "angle_deg"              # -> [sin(θ), cos(θ)]  (2 features)
  wave_elevation:
    type: "angle_deg"              # -> [sin(θ), cos(θ)]  (2 features)
  frequency_ghz:
    type: "scale"
    divisor: 10.0                  # -> freq / 10         (1 feature)
```

### Disabling global parameters

```yaml
# In datapipe config:
global_params_keys: []
global_params_normalization: {}

# In model config:
global_dim: null
```

---

## Training

### Single-GPU training

```bash
python train.py
```

### Multi-GPU training with `torchrun`

```bash
torchrun --nproc_per_node=8 train.py
```

### Monitoring with TensorBoard

```bash
tensorboard --logdir=outputs/tensorboard_logs
```

### Key metrics

| Metric                     | Meaning                                       |
|----------------------------|-----------------------------------------------|
| `train/loss`               | MSE in normalized log-space                   |
| `val/MSE`                  | Full-rollout MSE over all timesteps           |
| `val/timestep_X_MSE`       | Per-timestep MSE for drift analysis           |

### Output layout

Training outputs are saved to:

```text
stats/                             # Normalization statistics (auto-generated)
├── node_stats.json
├── feature_stats.json
└── geometry_stats.json

outputs/<date>/<time>/
├── checkpoints/                   # Model checkpoints (.mdlus and .pt)
└── tensorboard_logs/              # TensorBoard event files
```

The `stats/` directory is located at the launch directory and is reused
by the inference script automatically.

---

## Inference

Run inference on test cases:

```bash
python inference.py
```

The script automatically:

1. Discovers all `*_Animation` directories under the configured test
   path.
2. Loads training normalization statistics from `stats/`.
3. Verifies each test case has a `global_params.json` (if global
   parameters are enabled).
4. Runs time-conditional rollout ($T-1$ forward passes per case).
5. Denormalizes predictions: un-Z-score → `exp()` → physical $|E|$.
6. Saves VTU files with predicted, ground truth, and error fields.
7. Writes a PVD animation file for ParaView visualization.

### Output layout

```text
predictions/rank0/<case_name>/
├── geometry.vtp                    # Enclosure geometry
├── frame_001_pred.vtu              # Prediction at t = 1
├── frame_002_pred.vtu              # Prediction at t = 2
├── ...
├── frame_N_pred.vtu                # Prediction at t = N
├── global_params_used.json         # Provenance
└── prediction_animation.pvd        # ParaView time-animation
```

### VTU file contents

| Array                   | Description                            |
|-------------------------|----------------------------------------|
| `E_Magnitude_pred`      | Predicted $\|E\|$ (V/m)                |
| `E_Magnitude_exact`     | Ground truth $\|E\|$ (when available)  |
| `E_Magnitude_error`     | Signed error (pred − exact)            |
| `E_Magnitude_abs_error` | Absolute error                         |

---

## Architecture Details

### Data Flow

```text
STL file ──► unique vertices [M, 3] ──► normalize ──► geometry
                                                        │
VTU files ──► |E| [T, N, 1] ──► log(|E| + ε) ──► Z-score ──► features / targets
                                                                │
VTU coords ──► [N, 3] ──► normalize ──► coords                  │
                                                                │
Global params JSON ──► {sin/cos, scale} ──► global_features     │
                                                                │
             ┌──────────────────────────────────────────────────┘
             │
             ▼
    SimSample:
        node_features["coords"]   = [k, 3]     normalized positions
        node_features["features"] = [k, 1]     normalized log|E| at t=0
        node_features["geometry"] = [k_geo, 3] normalized STL positions
        node_features["time"]     = scalar     t / (T-1)
        global_features           = dict       encoded wave params
        node_target               = [k, 1]     normalized log|E| at time t
```

### Model Forward Pass

**Training mode** (one timestep per forward pass):

```text
coords [N, 3] ──┐
                ├─ FourierEncode ──┐
                │                  │
log|E|_0 [N, 1] ┼─────────────────►┼── concat ── [N, functional_dim] ──┐
                │                  │                                    │
t_norm [N, 1]  ─┘                  │                                    │
                                                                GeoTransolver
geometry [M, 3] ── (optional FourierEncode) ──► GALE attention ────────┤
                                                                        │
global_params ──► encode ──► [1, 1, G] ─────────────────────────────────┤
                                                                        │
                                                          pred [N, 1] ◄─┘
                                                         (log|E| at time t)
```

**Inference mode** (loops over all $T-1$ timesteps):

```python
for t in range(0, T - 1):
    sample.node_features["time"] = torch.tensor(t / (T - 1))
    pred_t = model.forward(sample, data_stats)        # [N, 1]
    predictions.append(pred_t)

full_rollout = torch.stack(predictions, dim=1)        # [N, T-1, 1]
```

### Fourier Features

The model supports two Fourier encoding strategies to mitigate spectral
bias in coordinate-based neural networks.

**Dyadic (NeRF-style):** deterministic geometric progression of
frequencies.

```math
\text{freqs} = \left[\, 2^0 \pi,\ 2^1 \pi,\ 2^2 \pi,\ \ldots,\ 2^{L-1} \pi \,\right]
```

For input $x$ and $L = 6$ frequencies, the encoding produces:

```math
\mathrm{FF}(x) = \bigl[\, x,\ \sin(\pi x),\ \cos(\pi x),\ \sin(2\pi x),\ \cos(2\pi x),\ \ldots,\ \sin(32\pi x),\ \cos(32\pi x) \,\bigr]
```

**Gaussian Random Fourier Features:** random projections sampled from
$\mathcal{N}(0, \sigma^2 I)$.

```math
B \sim \mathcal{N}(0, \sigma^2 I) \in \mathbb{R}^{L \times D}
```

```math
\mathrm{FF}(x) = \bigl[\, x,\ \sin(2\pi B x),\ \cos(2\pi B x) \,\bigr]
```

The random matrix $B$ is sampled once at model initialization (with a
fixed seed for reproducibility) and registered as a buffer, so it moves
with the model to the correct device and is included in checkpoints.

#### Recommended settings

| Use case                       | Mode       | `num_freqs` | `sigma` |
|--------------------------------|------------|-------------|---------|
| Default starting point         | `dyadic`   | 6           | —       |
| Sharp boundary features        | `dyadic`   | 8–10        | —       |
| Smoother functions             | `gaussian` | 64–128      | 10      |
| Very high-frequency content    | `gaussian` | 256         | 25      |

### Log-Transform Preprocessing

Raw $|E|$ values from EMA3D simulations typically exhibit:

- Range: 0 to ~200 V/m
- ~15–20% of points at or near zero (shielded regions)
- Heavy-tailed distribution unsuitable for direct MSE training

The preprocessing pipeline is:

```math
|E| \;\xrightarrow{\text{step 1}}\; \log\!\bigl(\max(|E|,\, 0) + \epsilon\bigr) \;\xrightarrow{\text{step 2}}\; \frac{\log(|E| + \epsilon) - \mu}{\sigma}
```

The inverse at inference:

```math
\hat{y}_{\text{norm}} \;\xrightarrow{\text{un-Z-score}}\; \hat{y}_{\log} \;\xrightarrow{\exp}\; \hat{y} - \epsilon \;\xrightarrow{\text{clamp}}\; |\hat{E}|_{\text{physical}}
```

The `exp()` operation **guarantees non-negative predictions**,
eliminating a common failure mode where the model produces physically
impossible negative field magnitudes.

The epsilon parameter `LOG_EPS` is chosen based on a quantitative
analysis of the data distribution. For most EMA3D datasets,
`LOG_EPS = 0.1` provides:

- `std / |mean|` ratio ≈ 1.95 (near-symmetric distribution)
- Dynamic range ≈ 8 in log-space (compact target space)
- ~18% of points clamped (only points with $|E| < 0.1$ V/m, which are
  below engineering significance for shielding problems)

---

## Project Structure

```text
oneshot_tcond/
├── conf/                                 # Hydra configuration files
│   ├── config.yaml                       # Main config (composes sub-configs)
│   ├── datapipe/
│   │   └── default.yaml                  # Dataset + sampling + global params
│   ├── inference/
│   │   └── default.yaml                  # Inference paths and options
│   ├── model/
│   │   ├── geotransolver_time_conditional.yaml
│   │   ├── transolver_autoregressive_rollout_training.yaml
│   │   └── transolver_one_step_rollout.yaml
│   ├── reader/
│   │   ├── d3plot.yaml
│   │   ├── vtp.yaml
│   │   └── zarr.yaml
│   └── training/
│       └── default.yaml                  # Epochs, LR, batch size, validation
├── datapipe.py                           # Dataset, SimSample, normalization
├── ema3d_reader.py                       # VTU / STL / global_params reader
├── rollout.py                            # GeoTransolverTimeConditionalEField model
├── train.py                              # Training script (Hydra + DDP)
├── inference.py                          # Inference script
├── analyze_efield_distribution.py        # Optional pre-training data analysis
├── readme.ipynb                          # Optional notebook walkthrough
├── stats/                                # Auto-generated normalization statistics (JSON)
└── outputs/                              # Hydra run directory: logs, checkpoints, predictions, TensorBoard
```

### Directory descriptions

| Path | Purpose |
|------|---------|
| `conf/` | Hydra configuration tree. Modify YAML files here to change behavior without touching code. |
| `datapipe.py` | Loads VTU + STL data, applies log-transform + Z-score, handles per-epoch random sampling, and builds `SimSample` objects. |
| `ema3d_reader.py` | Discovers `*_Animation` cases, parses STL geometry and VTU time series, and loads `global_params.json`. |
| `rollout.py` | Defines `GeoTransolverTimeConditionalEField` with time conditioning, Fourier feature encoding, and global parameter support. |
| `train.py` | Time-conditional training loop with distributed support, AMP, cosine LR schedule, and validation. |
| `inference.py` | Discovers test cases, runs full rollout, denormalizes predictions, and writes VTU + PVD files. |
| `stats/` | Auto-generated normalization statistics shared between training and inference. Delete to force recomputation. |
| `outputs/` | Hydra's run directory. Contains logs, model checkpoints, TensorBoard events, and prediction VTU files. |

---

## Results

On a dataset of 20 training cases and 2 validation cases with 125,000
field points per simulation and 20 timesteps per simulation, the model
achieves:

| Quantity                               | Value                          |
|----------------------------------------|--------------------------------|
| Training MSE (normalized log-space)    | ~0.05 after 1000 epochs        |
| Physical $\|E\|$ accuracy               | within ~1.3× of ground truth on average |
| Inference time per timestep (A100)     | ~120 ms                        |
| Full 20-step rollout (incl. VTU write) | ~2 s                           |

Early timesteps ($t = 0\text{–}2$), corresponding to the sharp
wave-penetration moment, exhibit higher error than steady-state
timesteps. This is expected behavior and improves with larger training
sets.

---

## Tuning Guide

### Addressing overfitting

If validation loss plateaus or increases while training loss continues
to decrease:

1. Reduce model capacity (`n_layers`, `n_hidden`, `slice_num`).
2. Increase `weight_decay` from `1e-4` to `1e-3`.
3. Acquire more training data (most impactful).

### Addressing underfitting

If training loss plateaus at a high value:

1. Enable Fourier features (`use_fourier: true`).
2. Increase model capacity.
3. Increase `start_lr` to `1e-4` or `2e-4`.
4. Verify `LOG_EPS` is appropriate via
   `analyze_efield_distribution.py`.
5. Confirm normalization statistics are in log-space.

### Working with large meshes

For simulations with more than 125k field points, use per-epoch random
sampling:

```yaml
# In datapipe config:
resolution: 50000
geometry_resolution: 20000
```

The Poisson-disc sampler draws a different random subset at each
`__getitem__` call, so over many epochs the model sees all points with
approximately uniform coverage. This works for $N > 2^{24}$, where
`torch.multinomial` fails.

---

## References

- Tancik, M., et al. *Fourier Features Let Networks Learn High Frequency
  Functions in Low Dimensional Domains.* NeurIPS 2020.
  [arxiv.org/abs/2006.10739](https://arxiv.org/abs/2006.10739)

- Wu, H., et al. *Transolver: A Fast Transformer Solver for PDEs on
  General Geometries.* ICML 2024.
  [arxiv.org/abs/2402.02366](https://arxiv.org/abs/2402.02366)

- Mildenhall, B., et al. *NeRF: Representing Scenes as Neural Radiance
  Fields for View Synthesis.* ECCV 2020.
  [arxiv.org/abs/2003.08934](https://arxiv.org/abs/2003.08934)

- PhysicsNeMo Crash Simulation Recipe:
  [github.com/NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo/tree/main/examples/structural_mechanics/crash)

- Ansys EMA3D: [ansys.com](https://www.ansys.com/)
