"""Autoregressive T-step rollout for WAMCast.

Direct port of scripts/uq/multihorizon_rollout_t16.py:166-278, restructured
for single-launch inference (batch dim = 1) and optional ensemble
aggregation.

Lag convention (matches multihorizon_rollout_t16.py:178-191, 228-233
exactly — verified against the reference source, not paraphrased):
  - lag48 is read from H5 truth at every rollout step k (1..ROLLOUT_STEPS):
    frame = launch_frame + (k - 17) * DELAY_FRAMES. This is always a
    strictly *past* frame relative to launch (k - 17 <= -1), so lag48 is
    fully teacher-forced for the whole rollout.
  - lag24 is read from H5 truth for k <= 9: frame = launch_frame +
    (k - 9) * DELAY_FRAMES (k=9 falls back to the launch frame itself).
    For k >= 10, lag24 is fed from the model's own trajectory: traj[k - 9].
  Note the fall-back index is k - 9 (not k - 8): the reference script's
  condition is `if k <= 9: ... else: lag24 = traj[k - 9]`. An off-by-one
  variant (k <= 8 / traj[k - 8]) appears in an earlier draft of this task
  but does not match the checked-in reference script and was not used here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import h5py
import numpy as np
import pandas as pd
import torch

from wamcast.dataset import LAG_PER_HOUR, LEV_HI, LEV_LO, ForecastInputs, _build_aux
from wamcast.drivers import DriverSeries
from wamcast.model import WAMCastModel

PathLike = Union[str, Path]

STEP_HOURS = 3                                  # hours advanced per rollout step
DELAY_FRAMES = STEP_HOURS * LAG_PER_HOUR         # 18 frames = 3h at 10-min cadence
DEFAULT_HORIZONS_HOURS = (3, 6, 12, 24, 48)
LAG24_TEACHER_FORCE_STEPS = 9                    # k <= 9 -> real H5 lag24
LAG48_STEP_OFFSET = 17                           # lag48 frame = launch + (k-17)*DELAY


@dataclass
class Forecast:
    mu: torch.Tensor                       # (H, B, C, lat, lon), z-scored
    sigma: torch.Tensor                    # (H, B, C, lat, lon), z-scored
    horizons_hours: tuple[int, ...]
    launch_utc: pd.Timestamp


def _load_norm_stats(stats_path: PathLike) -> tuple[np.ndarray, np.ndarray]:
    stats = np.load(stats_path)
    mean = np.where(np.isfinite(stats["mean"][LEV_LO:LEV_HI]),
                    stats["mean"][LEV_LO:LEV_HI], 0.0).astype(np.float32)
    std = np.where(np.isfinite(stats["std"][LEV_LO:LEV_HI]) &
                   (stats["std"][LEV_LO:LEV_HI] > 0),
                   stats["std"][LEV_LO:LEV_HI], 1.0).astype(np.float32)
    return mean[:, None, None], std[:, None, None]


def _read_rho_z(f: h5py.File, frame: int, mean: np.ndarray,
                 std: np.ndarray) -> torch.Tensor:
    r = f["rho"][int(frame), LEV_LO:LEV_HI, 1:-1, :]
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    z = (r - mean) / std
    return torch.from_numpy(z.astype(np.float32)).unsqueeze(0)


def _build_aux_at_frame(
    f: h5py.File, lat_grid: np.ndarray, lon_grid: np.ndarray,
    frame: int, drivers: DriverSeries,
) -> torch.Tensor:
    doy = float(f["doy"][int(frame)])
    at_time = pd.to_datetime(int(f["time"][int(frame)]), utc=True)
    aux_np = _build_aux(lat_grid, lon_grid, doy, at_time, drivers)
    return torch.from_numpy(aux_np).unsqueeze(0)


def rollout(
    model: WAMCastModel,
    inputs: ForecastInputs,
    *,
    drivers: DriverSeries,
    h5_path: PathLike,
    stats_path: PathLike,
    horizons_hours: Sequence[int] = DEFAULT_HORIZONS_HOURS,
) -> Forecast:
    """Single-member autoregressive rollout.

    Feed-forward convention (matches multihorizon_rollout_t16.py:220-245):
      - The 48h-offset lag is read from H5 truth for every step (teacher
        forced throughout the rollout window).
      - The 24h-offset lag is read from H5 truth for steps 1..9, and from
        the model's own trajectory for steps 10..T (see module docstring).
      - Auxiliary drivers are read at each step's source time
        (launch_frame + (k-1)*DELAY_FRAMES), matching build_aux's `shift`
        in the reference script.

    Returns a Forecast with sigma=0 (a single member carries no spread).
    """
    horizons_hours = tuple(horizons_hours)
    for h in horizons_hours:
        if h % STEP_HOURS != 0:
            raise ValueError(
                f"horizon {h}h is not a multiple of the {STEP_HOURS}h rollout step"
            )
    max_step = max(h // STEP_HOURS for h in horizons_hours)
    device = next(model.parameters()).device
    launch_frame = inputs.launch_frame

    with h5py.File(h5_path, "r") as f:
        lat_grid = f["lat"][1:-1]
        lon_grid = f["lon"][:]
        mean, std = _load_norm_stats(stats_path)

        real_lag48 = {}
        real_lag24 = {}
        aux_cache = {}
        for k in range(1, max_step + 1):
            real_lag48[k] = _read_rho_z(
                f, launch_frame + (k - LAG48_STEP_OFFSET) * DELAY_FRAMES, mean, std
            ).to(device)
            if k <= LAG24_TEACHER_FORCE_STEPS:
                real_lag24[k] = _read_rho_z(
                    f, launch_frame + (k - LAG24_TEACHER_FORCE_STEPS) * DELAY_FRAMES,
                    mean, std,
                ).to(device)
            aux_cache[k] = _build_aux_at_frame(
                f, lat_grid, lon_grid, launch_frame + (k - 1) * DELAY_FRAMES, drivers
            ).to(device)

    rho_cur = inputs.rho_cur.to(device)
    traj: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for k in range(1, max_step + 1):
            if k <= LAG24_TEACHER_FORCE_STEPS:
                lag24 = real_lag24[k]
            else:
                lag24 = traj[k - LAG24_TEACHER_FORCE_STEPS]
            lag48 = real_lag48[k]
            rho_lags = torch.stack([lag24, lag48], dim=1)
            pred = model.step(rho_cur, rho_lags, aux_cache[k])
            traj[k] = pred
            rho_cur = pred

    mu_stack = torch.stack(
        [traj[h // STEP_HOURS] for h in horizons_hours], dim=0
    )
    sigma_stack = torch.zeros_like(mu_stack)
    return Forecast(
        mu=mu_stack, sigma=sigma_stack,
        horizons_hours=horizons_hours, launch_utc=inputs.launch_utc,
    )


def ensemble_rollout(
    members: Sequence[WAMCastModel],
    inputs: ForecastInputs,
    *,
    drivers: DriverSeries,
    h5_path: PathLike,
    stats_path: PathLike,
    horizons_hours: Sequence[int] = DEFAULT_HORIZONS_HOURS,
) -> Forecast:
    """K-member ensemble rollout: independent rollouts, then aggregate.

    mu = ensemble mean, sigma = ensemble std (ddof=1, matching the sample
    standard deviation convention used elsewhere in the UQ pipeline).
    """
    horizons_hours = tuple(horizons_hours)
    if len(members) < 2:
        raise ValueError("ensemble_rollout needs at least 2 members to compute sigma")
    per_member = []
    for m in members:
        r = rollout(m, inputs, drivers=drivers, h5_path=h5_path,
                    stats_path=stats_path, horizons_hours=horizons_hours)
        per_member.append(r.mu.cpu())
    stack = torch.stack(per_member, dim=0)     # (K, H, B, C, lat, lon)
    mu = stack.mean(dim=0)
    sigma = stack.std(dim=0, unbiased=True).clamp(min=1e-9)
    return Forecast(
        mu=mu, sigma=sigma,
        horizons_hours=horizons_hours, launch_utc=inputs.launch_utc,
    )
