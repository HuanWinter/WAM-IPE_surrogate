"""NetCDF output for WAMCast forecasts (CF-1.10)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

PathLike = Union[str, Path]


@dataclass
class ForecastMetadata:
    launch_utc: pd.Timestamp
    driver_protocol: str          # "observed_omni" | "swpc_forecast" | "frozen"
    model_variant: str            # "standard_t16_ensemble_k20" | "sat_t16_perstep" | ...
    checkpoint_paths: tuple[str, ...]
    wamcast_version: str
    extra: dict = field(default_factory=dict)


def write_forecast_netcdf(
    path: PathLike,
    *,
    mu: np.ndarray,             # (H, B, C, lat, lon)
    sigma: np.ndarray,
    horizons_hours: tuple[int, ...],
    lat: np.ndarray,
    lon: np.ndarray,
    level: np.ndarray,
    metadata: ForecastMetadata,
    lo: Optional[np.ndarray] = None,
    hi: Optional[np.ndarray] = None,
    alpha: Optional[float] = None,
) -> None:
    """Write a WAMCast forecast to CF-1.10 NetCDF."""
    if mu.shape[1] != 1:
        raise ValueError(f"expected B=1 (single launch), got B={mu.shape[1]}")
    _validate_shape("mu", mu, len(horizons_hours), len(level), len(lat), len(lon))
    _validate_shape("sigma", sigma, len(horizons_hours), len(level), len(lat), len(lon))

    ds = xr.Dataset(
        data_vars={
            "rho_mu": (
                ("horizon", "level", "latitude", "longitude"),
                mu[:, 0].astype(np.float32),
                {"units": "z-score", "long_name": "z-scored mass density (mean)"},
            ),
            "rho_sigma": (
                ("horizon", "level", "latitude", "longitude"),
                sigma[:, 0].astype(np.float32),
                {"units": "z-score", "long_name": "z-scored mass density (std)"},
            ),
        },
        coords={
            "horizon": ("horizon", np.array(horizons_hours, dtype=np.int32),
                        {"units": "h", "long_name": "forecast horizon since launch"}),
            "level": ("level", level.astype(np.int32),
                      {"long_name": "WAM-IPE pressure level index"}),
            "latitude": ("latitude", lat.astype(np.float32),
                         {"units": "degrees_north", "standard_name": "latitude"}),
            "longitude": ("longitude", lon.astype(np.float32),
                          {"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.10",
            "title": "WAMCast thermospheric mass-density forecast",
            "launch_utc": metadata.launch_utc.isoformat(),
            "driver_protocol": metadata.driver_protocol,
            "model_variant": metadata.model_variant,
            "checkpoint_paths": ";".join(metadata.checkpoint_paths),
            "wamcast_version": metadata.wamcast_version,
            **metadata.extra,
        },
    )
    if lo is not None and hi is not None:
        _validate_shape("lo", lo, len(horizons_hours), len(level), len(lat), len(lon))
        _validate_shape("hi", hi, len(horizons_hours), len(level), len(lat), len(lon))
        if alpha is None:
            raise ValueError("alpha required when writing conformal intervals")
        ds["rho_lo"] = (
            ("horizon", "level", "latitude", "longitude"),
            lo[:, 0].astype(np.float32),
            {"units": "z-score", "long_name": "lower conformal bound"},
        )
        ds["rho_hi"] = (
            ("horizon", "level", "latitude", "longitude"),
            hi[:, 0].astype(np.float32),
            {"units": "z-score", "long_name": "upper conformal bound"},
        )
        ds.attrs["conformal_alpha"] = alpha
    ds.to_netcdf(path, engine="netcdf4")


def _validate_shape(name: str, arr: np.ndarray, H: int, C: int,
                    n_lat: int, n_lon: int) -> None:
    expected = (H, 1, C, n_lat, n_lon)
    if arr.shape != expected:
        raise ValueError(f"{name} has shape {arr.shape}, expected {expected}")
