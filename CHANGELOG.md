<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0a0] - 2025-08-XX

### Added

- Improved documentation for diffusion models and diffusion utils.
- Safe API to override `__init__`'s arguments saved in checkpoint file with
  `Module.from_checkpoint("chkpt.mdlus", override_args=set(...))`.
- PyTorch Geometric MeshGraphNet backend.
- Functionality in DoMINO to take arbitrary number of `scalar` or `vector`
  global parameters and encode them using `class ParameterModel`
- TopoDiff model and example.
- Diffusion models, metrics, and utils: implementation of Student-t
  distribution for EDM-based diffusion models (t-EDM). This feature is adapted
  from the paper [Heavy-Tailed Diffusion Models, Pandey et al.](https://arxiv.org/abs/2410.14171>).
  This includes a new EDM preconditioner (`tEDMPrecondSuperRes`), a loss
  function (`tEDMResidualLoss`), and a new option in corrdiff `diffusion_step`.
  :warning: This is an experimental feature that can be accessed through the
  `physicsnemo.experimental` module; it might also be subjected to API changes
  without notice.
- Bumped Ruff version from 0.0.290 to 0.12.5. Replaced Black with `ruff-format`.
- Domino improvements with Unet attention module and user configs

### Changed

- Diffusion utils: `physicsnemo.utils.generative` renamed into `physicsnemo.utils.diffusion`
- Diffusion models: in CorrDiff model wrappers (`EDMPrecondSuperResolution` and
  `UNet`), the arguments `profile_mode` and `amp_mode` cannot be overriden by
  `from_checkpoint`. They are now properties that can be dynamically changed
  *after* the model instantiation with, for example, `model.amp_mode = True`
  and `model.profile_mode = False`.
- Updated healpix data module to use correct `DistributedSampler` target for
  test data loader
- Existing DGL-based vortex shedding example has been renamed to `vortex_shedding_mgn_dgl`.
  Added new `vortex_shedding_mgn` example that uses PyTorch Geometric instead.
- HEALPixLayer can now use earth2grid HEALPix padding ops, if desired
- Migrated Vortex Shedding Reduced Mesh example to PyTorch Geometric.
- CorrDiff example: fixed bugs when training regression `UNet`.
- Diffusion models: fixed bugs related to gradient checkpointing on non-square
  images.
- Diffusion models: created a separate class `Attention` for clarity and
  modularity. Updated `UNetBlock` accordingly to use the `Attention` class
  instead of custom attention logic. This will update the model architecture
  for `SongUNet`-based diffusion models. Changes are not BC-breaking and are
  transparent to the user.
- :warning: **BC-breaking:** refactored the automatic mixed precision (AMP) API in layers
  and models defined in `physicsnemo/models/diffusion/` for improved usability.
  Note: it is now, not only possible, but *required* to explicitly set
  `model.amp_mode = True` in order to use the model in a `torch.autocast`
  clause. This applies to all `SongUNet`-based models.
- Diffusion models: fixed and improved API to enable fp16 forward pass in
  `UNet` and `EDMPrecondSuperResolution` model wrappers; fp16 forward pass can
  now be toggled/untoggled by setting `model.use_fp16 = True`.
- Diffusion models: improved API for Apex group norm. `SongUNet`-based models
  will automatically perform conversion of the input tensors to
  `torch.channels_last` memory format when `model.use_apex_gn` is `True`. New
  warnings are raised when attempting to use Apex group norm on CPU.
- Diffusion utils: systematic compilation of patching operations in `stochastic_sampler`
  for improved performance.
- Diffusion utils: patch-based inference and lead time support with deterministic sampler
- CorrDiff example: added option for Student-t EDM (t-EDM) in `train.py` and
  `generate.py`. When training a CorrDiff diffusion model, this feature can be
  enabled with the hydra overrides `++training.hp.distribution=student_t` and
  `++training.hp.nu_student_t=<nu_value>`. For generation, this feature can be
  enabled with similar overrides: `++generation.distribution=student_t` and
  `++generation.nu_student_t=<nu_value>`.
- CorrDiff example: the parameters `P_mean` and `P_std` (used to compute the
  noise level `sigma`) are now configurable. They can be set with the hydra
  overrides `++training.hp.P_mean=<P_mean_value>` and
  `++training.hp.P_std=<P_std_value>` for training (and similar ones with
  `training.hp` replaced by `generation` for generation).
- Diffusion utils: patch-based inference and lead time support with
  deterministic sampler.

### Deprecated

### Removed

### Fixed

### Security

### Dependencies

## [1.1.1] - 2025-06-16

### Fixed

- Fixed an inadvertent change to the deterministic sampler 2nd order correction
- Bug Fix in Domino model ball query layer

## [1.1.0] - 2025-06-05

### Added

- Added ReGen score-based data assimilation example
- General purpose patching API for patch-based diffusion
- New positional embedding selection strategy for CorrDiff SongUNet models
- Added Multi-Storage Client to allow checkpointing to/from Object Storage
- Added a new aerodynamics example using DoMINO to compute design sensitivities
  (e.g., drag adjoint) with respect to underlying input geometry.

### Changed

- Simplified CorrDiff config files, updated default values
- Refactored CorrDiff losses and samplers to use the patching API
- Support for non-square images and patches in patch-based diffusion
- ERA5 download example updated to use current file format convention and
  restricts global statistics computation to the training set
- Support for training custom StormCast models and various other improvements for StormCast
- Updated CorrDiff training code to support multiple patch iterations to amortize
  regression cost and usage of `torch.compile`
- Refactored `physicsnemo/models/diffusion/layers.py` to optimize data type
  casting workflow, avoiding unnecessary casting under autocast mode
- Refactored Conv2d to enable fusion of conv2d with bias addition
- Refactored GroupNorm, UNetBlock, SongUNet, SongUNetPosEmbd to support usage of
  Apex GroupNorm, fusion of activation with GroupNorm, and AMP workflow.
- Updated SongUNetPosEmbd to avoid unnecessary HtoD Memcpy of `pos_embd`
- Updated `from_checkpoint` to accommodate conversion between Apex optimized ckp
  and non-optimized ckp
- Refactored CorrDiff NVTX annotation workflow to be configurable
- Refactored `ResidualLoss` to support patch-accumlating training for
  amortizing regression costs
- Explicit handling of Warp device for ball query and sdf
- Merged SongUNetPosLtEmb with SongUNetPosEmb, add support for batch>1
- Add lead time embedding support for `positional_embedding_selector`. Enable
arbitrary positioning of probabilistic variables
- Enable lead time aware regression without CE loss
- Bumped minimum PyTorch version from 2.0.0 to 2.4.0, to minimize
  support surface for `physicsnemo.distributed` functionality.

### Dependencies

- Made `nvidia.dali` an optional dependency

## [1.0.1] - 2025-03-25

### Added

- Added version checks to ensure compatibility with older PyTorch for distributed
  utilities and ShardTensor

### Fixed

- `EntryPoint` error that occured during physicsnemo checkpoint loading

## [1.0.0] - 2025-03-18

### Added

- DoMINO model architecture, datapipe and training recipe
- Added matrix decomposition scheme to improve graph partitioning
- DrivAerML dataset support in FIGConvNet example.
- Retraining recipe for DoMINO from a pretrained model checkpoint
- Prototype support for domain parallelism of using ShardTensor (new).
- Enable DeviceMesh initialization via DistributedManager.
- Added Datacenter CFD use case.
- Add leave-in profiling utilities to physicsnemo, to easily enable torch/python/nsight
  profiling in all aspects of the codebase.

### Changed

- Refactored StormCast training example
- Enhancements and bug fixes to DoMINO model and training example
- Enhancement to parameterize DoMINO model with inlet velocity
- Moved non-dimensionaliztion out of domino datapipe to datapipe in domino example
- Updated utils in `physicsnemo.launch.logging` to avoid unnecessary `wandb` and `mlflow`
  imports
- Moved to experiment-based Hydra config in Lagrangian-MGN example
- Make data caching optional in `MeshDatapipe`
- The use of older `importlib_metadata` library is removed

### Deprecated

- ProcessGroupConfig is tagged for future deprecation in favor of DeviceMesh.

### Fixed

- Update pytests to skip when the required dependencies are not present
- Bug in data processing script in domino training example
- Fixed NCCL_ASYNC_ERROR_HANDLING deprecation warning

### Dependencies

- Remove the numpy dependency upper bound
- Moved pytz and nvtx to optional
- Update the base image for the Dockerfile
- Introduce Multi-Storage Client (MSC) as an optional dependency.
- Introduce `wrapt` as an optional dependency, needed when using
  ShardTensor's automatic domain parallelism

## [0.9.0] - 2024-12-04

### Added

- Graph Transformer processor for GraphCast/GenCast.
- Utility to generate STL from Signed Distance Field.
- Metrics for CAE and CFD domain such as integrals, drag, and turbulence invariances and
  spectrum.
- Added gradient clipping to StaticCapture utilities.
- Bistride Multiscale MeshGraphNet example.
- FIGConvUNet model and example.
- The Transolver model.
- The XAeroNet model.
- Incoporated CorrDiff-GEFS-HRRR model into CorrDiff, with lead-time aware SongUNet and
  cross entropy loss.
- Option to offload checkpoints to further reduce memory usage
- Added StormCast model training and simple inference to examples
- Multi-scale geometry features for DoMINO model.

### Changed

- Refactored CorrDiff training recipe for improved usability
- Fixed timezone calculation in datapipe cosine zenith utility.
- Refactored EDMPrecondSRV2 preconditioner and fixed the bug related to the metadata
- Extended the checkpointing utility to store metadata.
- Corrected missing export of loggin function used by transolver model

## [0.8.0] - 2024-09-24

### Added

- Graph Transformer processor for GraphCast/GenCast.
- Utility to generate STL from Signed Distance Field.
- Metrics for CAE and CFD domain such as integrals, drag, and turbulence invariances and
  spectrum.
- Added gradient clipping to StaticCapture utilities.
- Bistride Multiscale MeshGraphNet example.

### Changed

- Refactored CorrDiff training recipe for improved usability
- Fixed timezone calculation in datapipe cosine zenith utility.

## [0.7.0] - 2024-07-23

### Added

- Code logging for CorrDiff via Wandb.
- Augmentation pipeline for CorrDiff.
- Regression output as additional conditioning for CorrDiff.
- Learnable positional embedding for CorrDiff.
- Support for patch-based CorrDiff training and generation (stochastic sampling only)
- Enable CorrDiff multi-gpu generation
- Diffusion model for fluid data super-resolution (CMU contribution).
- The Virtual Foundry GraphNet.
- A synthetic dataloader for global weather prediction models, demonstrated on GraphCast.
- Sorted Empirical CDF CRPS algorithm
- Support for history, cos zenith, and downscaling/upscaling in the ERA5 HDF5 dataloader.
- An example showing how to train a "tensor-parallel" version of GraphCast on a
Shallow-Water-Equation example.
- 3D UNet
- AeroGraphNet example of training of MeshGraphNet on Ahmed body and DrivAerNet datasets.
- Warp SDF routine
- DLWP HEALPix model
- Pangu Weather model
- Fengwu model
- SwinRNN model
- Modulated AFNO model

### Changed

- Raise `PhysicsNeMoUndefinedGroupError` when querying undefined process groups
- Changed Indexing error in `examples/cfd/swe_nonlinear_pino` for `physicsnemo` loss function
- Safeguarding against uninitialized usage of `DistributedManager`

### Removed

- Remove mlflow from deployment image

### Fixed

- Fixed bug in the partitioning logic for distributing graph structures
intended for distributed message-passing.
- Fixed bugs for corrdiff diffusion training of `EDMv1` and `EDMv2`
- Fixed bug when trying to save DDP model trained through unified recipe

### Dependencies

- Update DALI to CUDA 12 compatible version.
- Update minimum python version to 3.10

## [0.6.0] - 2024-04-17

### Added

- The citation file.
- Link to the CWA dataset.
- ClimateDatapipe: an improved datapipe for HDF5/NetCDF4 formatted climate data
- Performance optimizations to CorrDiff.
- Physics-Informed Nonlinear Shallow Water Equations example.
- Warp neighbor search routine with a minimal example.
- Strict option for loading PhysicsNeMo checkpoints.
- Regression only or diffusion only inference for CorrDiff.
- Support for organization level model files on NGC file system
- Physics-Informed Magnetohydrodynamics example.

### Changed

- Updated Ahmed Body and Vortex Shedding examples to use Hydra config.
- Added more config options to FCN AFNO example.
- Moved posiitonal embedding in CorrDiff from the dataloader to network architecture

### Deprecated

- `physicsnemo.models.diffusion.preconditioning.EDMPrecondSR`. Use `EDMPecondSRV2` instead.

### Removed

- Pickle dependency for CorrDiff.

### Fixed

- Consistent handling of single GPU runs in DistributedManager
- Output location of objects downloaded with NGC file system
- Bug in scaling the conditional input in CorrDiff deterministic sampler

### Dependencies

- Updated DGL build in Dockerfile
- Updated default base image
- Moved Onnx from optional to required dependencies
- Optional Makani dependency required for SFNO model.

## [0.5.0] - 2024-01-25

### Added

- Distributed process group configuration mechanism.
- DistributedManager utility to instantiate process groups based on a process group config.
- Helper functions to faciliate distributed training with shared parameters.
- Brain anomaly detection example.
- Updated Frechet Inception Distance to use Wasserstein 2-norm with improved stability.
- Molecular Dynamics example.
- Improved usage of GraphPartition, added more flexible ways of defining a partitioned graph.
- Physics-Informed Stokes Flow example.
- Profiling markers, benchmarking and performance optimizations for CorrDiff inference.
- Unified weather model training example.

### Changed

- MLFLow logging such that only proc 0 logs to MLFlow.
- FNO given seperate methods for constructing lift and spectral encoder layers.

### Removed

- The experimental SFNO

### Dependencies

- Removed experimental SFNO dependencies
- Added CorrDiff dependencies (cftime, einops, pyspng, nvtx)
- Made tqdm a required dependency

## [0.4.0] - 2023-11-20

### Added

- Added Stokes flow dataset
- An experimental version of SFNO to be used in unified training recipe for
weather models
- Added distributed FFT utility.
- Added ruff as a linting tool.
- Ported utilities from PhysicsNeMo Launch to main package.
- EDM diffusion models and recipes for training and sampling.
- NGC model registry download integration into package/filesystem.
- Denoising diffusion tutorial.

### Changed

- The AFNO input argument `img_size` to `inp_shape`
- Integrated the network architecture layers from PhysicsNeMo-Sym.
- Updated the SFNO model, and the training and inference recipes.

### Fixed

- Fixed physicsnemo.Module `from_checkpoint` to work from custom model classes

### Dependencies

- Updated the base container to PyTorch 23.10.
- Updated examples to use Pydantic v2.

## [0.3.0] - 2023-09-21

### Added

- Added ability to compute CRPS(..., dim: int = 0).
- Added EFI for arbitrary climatological CDF.
- Added Kernel CRPS implementation (kcrps)
- Added distributed utilities to create process groups and orthogonal process groups.
- Added distributed AFNO model implementation.
- Added distributed utilities for communication of buffers of varying size per rank.
- Added distributed utilities for message passing across multiple GPUs.
- Added instructions for docker build on ARM architecture.
- Added batching support and fix the input time step for the DLWP wrapper.

### Changed

- Updating file system cache location to physicsnemo folder

### Fixed

- Fixed physicsnemo uninstall in CI docker image

### Security

- Handle the tar ball extracts in a safer way.

### Dependencies

- Updated the base container to latest PyTorch 23.07.
- Update DGL version.
- Updated require installs for python wheel
- Added optional dependency list for python wheel

## [0.2.1] - 2023-08-08

### Fixed

- Added a workaround fix for the CUDA graphs error in multi-node runs

### Security

- Update `certifi` package version

## [0.2.0] - 2023-08-07

### Added

- Added a CHANGELOG.md
- Added build support for internal DGL
- 4D Fourier Neural Operator model
- Ahmed body dataset
- Unified Climate Datapipe

### Changed

- DGL install changed from pypi to source
- Updated SFNO to add support for super resolution, flexible checkpoining, etc.

### Fixed

- Fixed issue with torch-harmonics version locking
- Fixed the PhysicsNeMo editable install
- Fixed AMP bug in static capture

### Security

- Fixed security issues with subprocess and urllib in `filesystem.py`

### Dependencies

- Updated the base container to latest PyTorch base container which is based on torch 2.0
- Container now supports CUDA 12, Python 3.10

## [0.1.0] - 2023-05-08

### Added

- Initial public release.
