# ema3d_reader.py
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
EMA3D VTU + STL Reader for Transient EM Simulation Data.

Reads simulation cases from directory structure:
    <case>_Animation/
        stl/  -> Static geometry mesh (STL) — geometry positions for GALE ball queries
        vtu/  -> Per-timestep E-field data (VTU) — local embeddings + positions

GeoTransolver / PhysicsNeMo mapping:
    local_embedding  = [E_Vector(3) + coords(3) + Fourier(6F)]  -> functional_dim
    local_positions  = field_coords [B, N, 3]                    -> local_positions (x_i)
    geometry         = STL positions [B, M, 3]                   -> geometry (ball query in GALE)

Note: PhysicsNeMo's ball_query.py requires geometry last dim = 3.
      The context_projector internally constructs relative positions
      and MLP features from the 3D coordinates. Normals are stored
      separately for reference/visualization but not passed to the model.
"""

import os
import re
import json   
import numpy as np
import pyvista as pv
from typing import Optional


# ==========================================
# FILE DISCOVERY
# ==========================================

def find_simulation_dirs(base_data_dir: str) -> list[str]:
    """Find all *_Animation simulation case directories, sorted naturally."""
    if not os.path.isdir(base_data_dir):
        return []

    dirs = [
        os.path.join(base_data_dir, d)
        for d in os.listdir(base_data_dir)
        if os.path.isdir(os.path.join(base_data_dir, d)) and d.endswith("_Animation")
    ]

    def natural_key(name):
        return [
            int(s) if s.isdigit() else s.lower()
            for s in re.findall(r"\d+|\D+", os.path.basename(name))
        ]

    return sorted(dirs, key=natural_key)


def find_vtu_files(vtu_dir: str) -> list[str]:
    """Find all VTU files in directory, sorted by frame number."""
    if not os.path.isdir(vtu_dir):
        return []

    vtu_files = [
        os.path.join(vtu_dir, f)
        for f in os.listdir(vtu_dir)
        if f.lower().endswith(".vtu") and not f.startswith(".")
    ]

    def frame_key(path):
        m = re.search(r"frame_(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else 0

    return sorted(vtu_files, key=frame_key)


def find_stl_file(stl_dir: str) -> Optional[str]:
    """Find the STL file in directory."""
    if not os.path.isdir(stl_dir):
        return None

    stl_files = [
        os.path.join(stl_dir, f)
        for f in os.listdir(stl_dir)
        if f.lower().endswith(".stl")
    ]

    return stl_files[0] if stl_files else None


def _load_global_params(case_dir: str) -> dict:
    """Load global params from <case_dir>/global_params.json if present."""
    path = os.path.join(case_dir, "global_params.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


# ==========================================
# STL GEOMETRY LOADER
# ==========================================

def load_stl_geometry(stl_path: str, debug: bool = False) -> dict:
    """
    Load STL geometry for GeoTransolver.

    Returns positions [M, 3] for the model's geometry input (ball queries),
    and normals [M, 3] separately for reference/visualization.

    PhysicsNeMo's ball_query.py enforces last_dim == 3, so only
    positions are passed to the model. The context_projector
    internally builds MLP features from the 3D coordinates.

    Args:
        stl_path: Path to STL geometry file
        debug: Enable verbose output

    Returns:
        Dictionary with:
            - "geometry_pos":     [M, 3] vertex positions (passed to model)
            - "geometry_normals": [M, 3] vertex normals (reference only)
    """
    raw_mesh = pv.read(stl_path)

    mesh = raw_mesh.extract_surface().clean()

    mesh.compute_normals(
        cell_normals=False,
        point_normals=True,
        inplace=True,
        consistent_normals=True,
        auto_orient_normals=True,
    )

    positions = np.array(mesh.points, dtype=np.float64)  # [M, 3]
    normals = np.array(mesh.point_data["Normals"], dtype=np.float64)  # [M, 3]

    # Normalize normals (safety)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normals = normals / norms

    if debug:
        print(f"    [STL] {os.path.basename(stl_path)}")
        print(f"      Raw triangles:     {raw_mesh.n_cells}")
        print(f"      Unique vertices M: {positions.shape[0]}")
        print(f"      geometry_pos:      {positions.shape}  (-> model geometry input)")
        print(f"      geometry_normals:  {normals.shape}  (-> reference only)")
        print(f"      Pos range: x[{positions[:, 0].min():.6f}, {positions[:, 0].max():.6f}]")
        print(f"                 y[{positions[:, 1].min():.6f}, {positions[:, 1].max():.6f}]")
        print(f"                 z[{positions[:, 2].min():.6f}, {positions[:, 2].max():.6f}]")

    return {
        "geometry_pos": positions,        # [M, 3] — for model
        "geometry_normals": normals,      # [M, 3] — for reference
    }


# ==========================================
# VTU FIELD LOADER
# ==========================================

# ema3d_reader.py
# Only showing changed functions — file discovery and STL loading unchanged

def load_vtu_time_series(
    vtu_files: list[str],
    debug: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Load VTU time series and extract field coordinates + E_Magnitude.

    Returns:
        Tuple of:
            - field_coords: [N, 3] grid coordinates (x_i)
            - fields: dict with:
                "E_Magnitude": [T, N, 1] electric field magnitude
    """
    if not vtu_files:
        raise FileNotFoundError("No VTU files provided")

    first_mesh = pv.read(vtu_files[0])
    field_coords = np.array(first_mesh.points, dtype=np.float64)
    N = field_coords.shape[0]

    available_keys = list(first_mesh.point_data.keys())

    if debug:
        print(f"    [VTU] Found {len(vtu_files)} frames")
        print(f"      Field nodes N: {N}")
        print(f"      Available keys: {available_keys}")
        print(f"      x_i range: x[{field_coords[:, 0].min():.6f}, {field_coords[:, 0].max():.6f}]")
        print(f"                 y[{field_coords[:, 1].min():.6f}, {field_coords[:, 1].max():.6f}]")
        print(f"                 z[{field_coords[:, 2].min():.6f}, {field_coords[:, 2].max():.6f}]")

    e_magnitudes = []

    for i, vtu_path in enumerate(vtu_files):
        mesh = pv.read(vtu_path)

        if mesh.n_points != N:
            raise ValueError(
                f"Node count mismatch in {os.path.basename(vtu_path)}: "
                f"got {mesh.n_points}, expected {N}"
            )

        if "E_Magnitude" in mesh.point_data:
            em = np.asarray(mesh.point_data["E_Magnitude"], dtype=np.float64)
            if em.ndim > 1:
                em = em.flatten()
            assert em.shape[0] == N
            e_magnitudes.append(em)
        else:
            raise ValueError(
                f"No E_Magnitude field found in {os.path.basename(vtu_path)}"
            )

    fields = {}

    # [T, N] -> [T, N, 1]
    fields["E_Magnitude"] = np.stack(e_magnitudes, axis=0)[:, :, np.newaxis]

    if debug:
        v = fields["E_Magnitude"]
        print(f"      E_Magnitude: shape={v.shape}, range=[{v.min():.6e}, {v.max():.6e}]")

    return field_coords, fields


def load_simulation_case(sim_dir: str, debug: bool = False) -> dict:
    """
    Load a complete simulation case.

    Returns:
        Dictionary containing:
            - "field_coords":      [N, 3]    -> local_positions
            - "geometry_pos":      [M, 3]    -> geometry (ball query)
            - "geometry_normals":  [M, 3]    -> reference only
            - "E_Magnitude":       [T, N, 1] -> feature / target
            - "global_params":     dict       -> wave angle, frequency, etc.
    """
    case_name = os.path.basename(sim_dir)

    if debug:
        print(f"\n    [CASE] {case_name}")

    stl_dir = os.path.join(sim_dir, "stl")
    vtu_dir = os.path.join(sim_dir, "vtu")

    # Load STL geometry
    stl_path = find_stl_file(stl_dir)
    if stl_path is None:
        raise FileNotFoundError(f"No STL file found in {stl_dir}")
    geometry = load_stl_geometry(stl_path, debug=debug)

    # Load VTU time series
    vtu_files = find_vtu_files(vtu_dir)
    if not vtu_files:
        raise FileNotFoundError(f"No VTU files found in {vtu_dir}")
    field_coords, fields = load_vtu_time_series(vtu_files, debug=debug)

    # ── Load global parameters from JSON ──
    global_params = _load_global_params(sim_dir)

    M = geometry["geometry_pos"].shape[0]
    N = field_coords.shape[0]
    T = fields["E_Magnitude"].shape[0]

    if debug:
        print(f"      Summary: M={M} geometry pts, N={N} field pts, T={T} timesteps")
        if global_params:
            print(f"      [GLOBAL] {global_params}")
        else:
            print(f"      [GLOBAL] No global_params.json found (using empty dict)")

    record = {
        "field_coords": field_coords,
        "geometry_pos": geometry["geometry_pos"],
        "geometry_normals": geometry["geometry_normals"],
        "global_params": global_params,        # ← ADD
        **fields,
    }

    return record




# ==========================================
# BATCH PROCESSOR
# ==========================================

def process_ema3d_data(
    data_dir: str,
    num_samples: Optional[int] = None,
    logger=None,
    debug: bool = False,
) -> list[dict]:
    """Process all simulation cases in a directory."""
    sim_dirs = find_simulation_dirs(data_dir)

    if not sim_dirs:
        msg = f"No *_Animation directories found in: {data_dir}"
        if logger:
            logger.error(msg)
        print(f"ERROR: {msg}")
        return []

    if debug:
        print(f"\n{'=' * 60}")
        print(f"EMA3D Data Processing (GeoTransolver / GALE)")
        print(f"{'=' * 60}")
        print(f"  Directory:  {data_dir}")
        print(f"  Found:      {len(sim_dirs)} simulation cases")
        print(f"  Requested:  {num_samples or 'all'}")

    data_records = []

    for i, sim_dir in enumerate(sim_dirs):
        if num_samples is not None and i >= num_samples:
            break

        case_name = os.path.basename(sim_dir)

        if logger:
            logger.info(f"Processing: {case_name}")

        try:
            record = load_simulation_case(sim_dir, debug=debug)
            data_records.append(record)
        except Exception as e:
            msg = f"Error processing {case_name}: {e}"
            if logger:
                logger.error(msg)
            print(f"ERROR: {msg}")
            continue

    if debug:
        print(f"\n  Successfully processed {len(data_records)} / {len(sim_dirs)} cases")
        if data_records:
            rec = data_records[0]
            print(f"  First record:")
            for k, v in rec.items():
                if hasattr(v, "shape"):
                    print(f"    {k:20s}: shape={str(v.shape):15s} dtype={v.dtype}")
                elif isinstance(v, dict):
                    print(f"    {k:20s}: dict with keys={list(v.keys())}")
                else:
                    print(f"    {k:20s}: {type(v).__name__} = {v}")
        print(f"{'=' * 60}\n")


    return data_records


# ==========================================
# READER CLASS
# ==========================================

class Reader:
    """
    EMA3D VTU+STL Reader for GeoTransolver with GALE.

    Output record -> GeoTransolver.forward() mapping:
        field_coords      [N, 3]  -> local_positions
        E_Vector[t]       [N, 3]  -> local_embedding
        geometry_pos      [M, 3]  -> geometry  (ball_query requires dim=3)
        geometry_normals  [M, 3]  -> reference only (not passed to model)

    Usage:
        reader = Reader(debug=True)
        records = reader(data_dir="./data", num_samples=10, split="train")
    """

    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(
        self,
        data_dir: str,
        num_samples: int,
        split: Optional[str] = None,
        logger=None,
        **kwargs,
    ) -> list[dict]:
        if self.debug and split:
            print(f"\n  [Reader] Loading split='{split}' from {data_dir}")

        return process_ema3d_data(
            data_dir=data_dir,
            num_samples=num_samples,
            logger=logger,
            debug=self.debug,
        )


if __name__ == "__main__":
    BASE_DIR = "/workspace/aghasemi/isv/ema3d/data/raw_hdf5/vtr_vtu_output"

    reader = Reader(debug=True)
    records = reader(data_dir=BASE_DIR, num_samples=2, split="test")

    if records:
        print("\n--- Loaded Records Summary ---")
        for i, rec in enumerate(records):
            print(f"\n  Case {i}:")
            for k, v in rec.items():
                print(f"    {k:20s}: shape={str(v.shape):15s} dtype={v.dtype}")

        rec = records[0]
        print(f"\n--- GeoTransolver Config ---")
        print(f"  geometry_dim   = 3   (positions only, ball_query requires dim=3)")
        print(f"  functional_dim = 3 + 3 + 6F  (E_Vector + coords + Fourier)")
        print(f"  out_dim        = 3   (predict dEx, dEy, dEz)")
    else:
        print("No records loaded.")
