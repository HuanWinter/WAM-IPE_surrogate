"""Driver ingest: OMNI2 (observed), SWPC RSGA (forecast), frozen (ablation).

All three sources produce a DriverSeries on the model's 10-minute time grid
(same cadence as the ML-ready H5). The rollout code uses (Kp, F10.7) at the
target time of each rollout step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]
GRID_FREQ = "10min"


@dataclass
class DriverSeries:
    """(Kp, F10.7) time series on the 10-minute grid."""
    time: pd.DatetimeIndex
    kp: np.ndarray      # shape (N,), float32
    f107: np.ndarray    # shape (N,), float32

    def __post_init__(self) -> None:
        if not (len(self.time) == len(self.kp) == len(self.f107)):
            raise ValueError(
                f"length mismatch: time={len(self.time)}, "
                f"kp={len(self.kp)}, f107={len(self.f107)}"
            )
        self.kp = self.kp.astype(np.float32)
        self.f107 = self.f107.astype(np.float32)

    def as_h5_driver_array(self) -> np.ndarray:
        """Return (N, 2) float32 array with columns [Kp, F10.7] matching the
        H5 sidecar convention used by the training pipeline
        (see scripts/backfill_drivers_2025.py:103 in the research repo)."""
        return np.stack([self.kp, self.f107], axis=1).astype(np.float32)


def _grid(start, end) -> pd.DatetimeIndex:
    """Build a 10-min UTC-aware DatetimeIndex spanning [start, end] inclusive.

    Accepts either UTC-aware or tz-naive inputs; naive inputs are localized
    to UTC (they are assumed to already be UTC — the H5 grid is UTC).
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if s.tz is None:
        s = s.tz_localize("UTC")
    if e.tz is None:
        e = e.tz_localize("UTC")
    return pd.date_range(s, e, freq=GRID_FREQ, tz="UTC")


def load_omni_csv(path: PathLike, start: str, end: str) -> DriverSeries:
    """Load OMNI2 Kp/F10.7 from a CSV with columns time_utc,Kp,F107.

    Values sparser than 10 min (3-hour Kp, daily F10.7) are forward-filled
    onto the 10-min grid.
    """
    df = pd.read_csv(path, parse_dates=["time_utc"])
    df = df.set_index("time_utc").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    g = _grid(start, end)
    kp = df["Kp"].reindex(g, method="ffill").to_numpy()
    f107 = df["F107"].reindex(g, method="ffill").to_numpy()
    return DriverSeries(time=g, kp=kp, f107=f107)


def load_swpc_rsga_json(path: PathLike, start: str, end: str) -> DriverSeries:
    """Load SWPC RSGA forecast bulletin (JSON schema documented in
    docs/input-formats.md). Each daily Kp / F10.7 is held constant over its
    UT day."""
    with open(path) as f:
        b = json.load(f)
    g = _grid(start, end)
    kp = np.full(len(g), np.nan, dtype=np.float32)
    f107 = np.full(len(g), np.nan, dtype=np.float32)
    for row in b["kp_forecast_daily"]:
        day_start = pd.Timestamp(row["date"], tz="UTC")
        day_end = day_start + pd.Timedelta("1D")
        mask = (g >= day_start) & (g < day_end)
        kp[mask] = float(row["kp_max"])
    for row in b["f107_forecast_daily"]:
        day_start = pd.Timestamp(row["date"], tz="UTC")
        day_end = day_start + pd.Timedelta("1D")
        mask = (g >= day_start) & (g < day_end)
        f107[mask] = float(row["f107"])
    if np.isnan(kp).all() or np.isnan(f107).all():
        raise ValueError(
            f"SWPC bulletin at {path} covers "
            f"{[row['date'] for row in b['kp_forecast_daily']]} but requested "
            f"window [{start}, {end}] has zero overlap — nothing to fill"
        )
    # Any leading NaNs before the forecast start get filled from the first valid.
    kp_series = pd.Series(kp).bfill().ffill().to_numpy()
    f107_series = pd.Series(f107).bfill().ffill().to_numpy()
    return DriverSeries(time=g, kp=kp_series, f107=f107_series)


def frozen_drivers(kp: float, f107: float, start: str, end: str) -> DriverSeries:
    """Driver-frozen ablation protocol: constant Kp/F10.7 across the window."""
    g = _grid(start, end)
    return DriverSeries(
        time=g,
        kp=np.full(len(g), kp, dtype=np.float32),
        f107=np.full(len(g), f107, dtype=np.float32),
    )
