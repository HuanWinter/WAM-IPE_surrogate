"""Tests for NetCDF writer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wamcast.io import ForecastMetadata, write_forecast_netcdf


@pytest.fixture
def dummy_forecast_arrays():
    H, B, C, lat, lon = 5, 1, 41, 89, 90
    return dict(
        mu=np.random.randn(H, B, C, lat, lon).astype(np.float32),
        sigma=np.abs(np.random.randn(H, B, C, lat, lon)).astype(np.float32) + 0.1,
        horizons_hours=(3, 6, 12, 24, 48),
        lat=np.linspace(-88, 88, lat).astype(np.float32),
        lon=np.linspace(0, 358, lon).astype(np.float32),
        level=np.arange(10, 51, dtype=np.int32),
    )


def test_write_forecast_netcdf_creates_valid_cf_file(tmp_path, dummy_forecast_arrays):
    out = tmp_path / "forecast.nc"
    meta = ForecastMetadata(
        launch_utc=pd.Timestamp("2025-11-11T00:00:00Z"),
        driver_protocol="swpc_forecast",
        model_variant="standard_t16_ensemble_k20",
        checkpoint_paths=("member_00/best.ckpt", "member_01/best.ckpt"),
        wamcast_version="0.1.0",
    )
    write_forecast_netcdf(out, **dummy_forecast_arrays, metadata=meta)

    ds = xr.open_dataset(out)
    assert "rho_mu" in ds.data_vars
    assert "rho_sigma" in ds.data_vars
    assert ds["rho_mu"].dims == ("horizon", "level", "latitude", "longitude")
    assert ds.attrs["driver_protocol"] == "swpc_forecast"
    assert ds.attrs["wamcast_version"] == "0.1.0"
    assert ds.attrs["Conventions"] == "CF-1.10"
    assert "2025-11-11" in ds.attrs["launch_utc"]


def test_write_forecast_netcdf_with_intervals(tmp_path, dummy_forecast_arrays):
    out = tmp_path / "forecast_intervals.nc"
    lo = dummy_forecast_arrays["mu"] - 1.5 * dummy_forecast_arrays["sigma"]
    hi = dummy_forecast_arrays["mu"] + 1.5 * dummy_forecast_arrays["sigma"]
    meta = ForecastMetadata(
        launch_utc=pd.Timestamp("2025-11-11T00:00:00Z"),
        driver_protocol="observed_omni",
        model_variant="standard_t16_ensemble_k20",
        checkpoint_paths=("m.ckpt",),
        wamcast_version="0.1.0",
    )
    write_forecast_netcdf(out, **dummy_forecast_arrays,
                          lo=lo, hi=hi, alpha=0.05, metadata=meta)

    ds = xr.open_dataset(out)
    assert "rho_lo" in ds.data_vars
    assert "rho_hi" in ds.data_vars
    assert ds.attrs["conformal_alpha"] == 0.05


def test_write_forecast_rejects_batch_gt_1(tmp_path, dummy_forecast_arrays):
    """Single-launch inference; batch dim must be 1 at write time."""
    dummy_forecast_arrays["mu"] = np.random.randn(5, 2, 41, 89, 90).astype(np.float32)
    dummy_forecast_arrays["sigma"] = np.ones((5, 2, 41, 89, 90), dtype=np.float32)
    meta = ForecastMetadata(
        launch_utc=pd.Timestamp("2025-11-11T00:00:00Z"),
        driver_protocol="frozen",
        model_variant="standard_t16_single",
        checkpoint_paths=("m.ckpt",),
        wamcast_version="0.1.0",
    )
    with pytest.raises(ValueError, match="B=1"):
        write_forecast_netcdf(tmp_path / "bad.nc", **dummy_forecast_arrays, metadata=meta)
