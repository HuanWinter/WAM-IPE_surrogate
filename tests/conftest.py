"""Shared test fixtures."""
from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pandas as pd
import pytest


TESTS_DIR = pathlib.Path(__file__).parent


@pytest.fixture(scope="session")
def demo_h5(tmp_path_factory) -> pathlib.Path:
    """Build a 5-day, 41-level, 91x90 H5 slice around 2025-11-11 super-storm.

    Starts 2025-11-09T00:00Z (48h before the storm) so a launch at the
    storm's onset (2025-11-11T00:00Z, tested below) has the full 48h lag
    window available before it, matching the real launch-readiness check
    in storm_subset() (scripts/uq/multihorizon_rollout_t16.py:78-81):
    `idx >= MAX_LAG_STEPS` where MAX_LAG_STEPS = 48h * 6 frames/h.

    5 days (not 3) so a T=16 autoregressive rollout from the 2025-11-11T00Z
    launch (frame 288) has room for its aux/lag reads, which reach as far
    as launch_frame + 15*18 = 558 frames ahead (see rollout._build_aux_at_frame
    at the last rollout step) — comfortably inside 720 frames but outside
    the old 432-frame (3-day) span.

    Not a physically meaningful synthetic - just structurally identical to
    ML_ready_23-26_clean.h5 so dataset assembly can be tested end-to-end.
    """
    out = tmp_path_factory.mktemp("h5") / "demo_launch.h5"
    n_frames = 5 * 24 * 6  # 5 days * 24 h * 6 frames/h = 720
    n_lev = 51
    n_lat = 91
    n_lon = 90
    start = pd.Timestamp("2025-11-09T00:00:00Z")
    times = pd.date_range(start, periods=n_frames, freq="10min", tz="UTC")
    rng = np.random.default_rng(42)
    rho = rng.uniform(1e-13, 5e-12, (n_frames, n_lev, n_lat, n_lon)).astype(np.float32)
    with h5py.File(out, "w") as f:
        f.create_dataset("rho", data=rho, compression="gzip", compression_opts=1)
        f.create_dataset("lat", data=np.linspace(-90, 90, n_lat).astype(np.float32))
        f.create_dataset("lon", data=np.linspace(0, 358, n_lon).astype(np.float32))
        # .as_unit("ns") is required: pandas>=2.2 date_range() defaults to
        # microsecond resolution, so a bare .asi8 here would silently write
        # microsecond-scale ints mislabeled as nanoseconds. The real H5
        # sidecar (WAM-IPE/Res/ML_ready_23-26_clean.h5) stores true int64 ns
        # since epoch (verified: 1688159400000000000 == 2023-06-30T21:10:00Z).
        f.create_dataset("time", data=times.as_unit("ns").asi8)  # int64 nanoseconds
        f.create_dataset("doy", data=np.array(
            [t.dayofyear + t.hour / 24 for t in times], dtype=np.float32))
        # Column order MUST be [Kp, F10.7] - matches the trained model's H5
        # sidecar convention (backfill_drivers_2025.py:103 in the research repo).
        # Swapping these silently produces physically wrong forecasts with no
        # crash and no test signal.
        f.create_dataset("driver", data=np.stack([
            np.full(n_frames, 5.0, dtype=np.float32),    # Kp (col 0)
            np.full(n_frames, 168.0, dtype=np.float32),  # F10.7 (col 1)
        ], axis=1))
    return out


@pytest.fixture(scope="session")
def demo_stats(tmp_path_factory) -> pathlib.Path:
    """Per-level mean/std for the demo H5 (matches training stats schema)."""
    out = tmp_path_factory.mktemp("stats") / "demo_stats.npz"
    np.savez(out,
             mean=np.full(51, -28.5, dtype=np.float32),
             std=np.full(51, 1.2, dtype=np.float32))
    return out
