"""Wall-clock launch-time input assembly for WAMCast forecasts.

The training-side UpperMultiLagDataset (train_camnet_multilag.py:149-232)
is indexed by integer H5 frame - that convention makes sense inside the
research pipeline but is hostile to a user who just wants to forecast
"starting at 2025-11-11T00Z, using the drivers I have on hand." This
module wraps the same tensor-assembly logic behind a from_launch(utc)
factory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pandas as pd
import torch

from wamcast.drivers import DriverSeries

PathLike = Union[str, Path]

LAG_HOURS: tuple[int, int] = (24, 48)   # matches train_camnet_multilag.py LAGS
LAG_PER_HOUR: int = 6                    # 10-min grid -> 6 frames per hour
LEV_LO, LEV_HI = 10, 51                  # matches train_camnet_upper.py
RHO_CH = LEV_HI - LEV_LO                 # 41 levels used by the model


@dataclass
class ForecastInputs:
    """(rho_cur, rho_lags, aux) ready for WAMCastModel.step. Batch dim = 1."""
    rho_cur: torch.Tensor
    rho_lags: torch.Tensor
    aux: torch.Tensor
    launch_utc: pd.Timestamp
    launch_frame: int   # index into the H5 time axis

    @classmethod
    def from_launch(
        cls,
        h5_path: PathLike,
        stats_path: PathLike,
        launch_utc: str,
        drivers: DriverSeries,
    ) -> "ForecastInputs":
        launch = pd.Timestamp(launch_utc)
        if launch.tz is None:
            launch = launch.tz_localize("UTC")

        with h5py.File(h5_path, "r") as f:
            time_ns = f["time"][:]
            times = pd.to_datetime(time_ns, utc=True)
            frame = int(np.searchsorted(time_ns, launch.value))
            if not (0 <= frame < len(times)) or times[frame] != launch:
                raise ValueError(
                    f"launch {launch_utc} not on H5 time grid; "
                    f"nearest = {times[max(0, frame - 1):frame + 2].tolist()}"
                )
            max_lag_frames = max(LAG_HOURS) * LAG_PER_HOUR
            if frame < max_lag_frames:
                raise ValueError(
                    f"launch {launch_utc} is before the model's lag window "
                    f"(need {max_lag_frames} prior frames; H5 has {frame})"
                )
            stats = np.load(stats_path)
            mean = np.where(np.isfinite(stats["mean"][LEV_LO:LEV_HI]),
                            stats["mean"][LEV_LO:LEV_HI], 0.0).astype(np.float32)
            std = np.where(np.isfinite(stats["std"][LEV_LO:LEV_HI]) &
                           (stats["std"][LEV_LO:LEV_HI] > 0),
                           stats["std"][LEV_LO:LEV_HI], 1.0).astype(np.float32)
            m3 = mean[:, None, None]
            s3 = std[:, None, None]

            def _read(i: int) -> np.ndarray:
                r = f["rho"][int(i), LEV_LO:LEV_HI, 1:-1, :]
                r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
                return ((r - m3) / s3).astype(np.float32)

            rho_cur_np = _read(frame)
            lag_frames = [frame - h * LAG_PER_HOUR for h in LAG_HOURS]
            rho_lags_np = np.stack([_read(i) for i in lag_frames], axis=0)

            lat_grid = f["lat"][1:-1]
            lon_grid = f["lon"][:]
            doy = float(f["doy"][frame])

        aux_np = _build_aux(lat_grid, lon_grid, doy, launch, drivers)

        return cls(
            rho_cur=torch.from_numpy(rho_cur_np).unsqueeze(0),
            rho_lags=torch.from_numpy(rho_lags_np).unsqueeze(0),
            aux=torch.from_numpy(aux_np).unsqueeze(0),
            launch_utc=launch,
            launch_frame=frame,
        )


def _build_aux(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    doy: float,
    at_time: pd.Timestamp,
    drivers: DriverSeries,
) -> np.ndarray:
    """Assemble the 10-channel aux tensor. Order matches
    train_camnet_multilag.py:203-204 (driver[0]=Kp, driver[1]=F10.7) and
    scripts/uq/multihorizon_rollout_t16.py:build_aux."""
    # searchsorted directly on the DatetimeIndex (not via .asi8/.value) so the
    # lookup is correct regardless of the index's internal time resolution.
    # pandas>=2.2 DatetimeIndex.asi8 returns ints in the index's *native*
    # unit (often microseconds for date_range output), not nanoseconds, so
    # comparing it against a nanosecond scalar like Timestamp.value silently
    # produces a ~1000x-wrong lookup.
    driver_idx = int(drivers.time.searchsorted(at_time))
    driver_idx = min(driver_idx, len(drivers.time) - 1)
    kp = float(drivers.kp[driver_idx])
    f107 = float(drivers.f107[driver_idx])
    hour = at_time.hour + at_time.minute / 60.0
    lon_g, lat_g = np.meshgrid(lon_grid, lat_grid)
    return np.stack([
        np.sin(np.radians(lat_g)), np.cos(np.radians(lat_g)),
        np.sin(np.radians(lon_g)), np.cos(np.radians(lon_g)),
        np.sin(2 * np.pi * doy / 365.25) * np.ones_like(lat_g),
        np.cos(2 * np.pi * doy / 365.25) * np.ones_like(lat_g),
        np.sin(2 * np.pi * hour / 24.0) * np.ones_like(lat_g),
        np.cos(2 * np.pi * hour / 24.0) * np.ones_like(lat_g),
        np.full_like(lat_g, kp),      # channel 8 = Kp (matches trained model)
        np.full_like(lat_g, f107),    # channel 9 = F10.7
    ], axis=0).astype(np.float32)
