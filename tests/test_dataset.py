"""Tests for ForecastInputs (wall-clock launch-time API)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from wamcast.dataset import ForecastInputs, LAG_HOURS
from wamcast.drivers import frozen_drivers


def test_forecast_inputs_shape(demo_h5, demo_stats):
    drivers = frozen_drivers(kp=6.7, f107=168.0,
                             start="2025-11-10T00:00:00Z",
                             end="2025-11-13T00:00:00Z")
    inputs = ForecastInputs.from_launch(
        h5_path=demo_h5,
        stats_path=demo_stats,
        launch_utc="2025-11-11T00:00:00Z",
        drivers=drivers,
    )
    # Single-launch inference - batch dim = 1
    assert inputs.rho_cur.shape == (1, 41, 89, 90)
    assert inputs.rho_lags.shape == (1, 2, 41, 89, 90)  # 24h + 48h lags
    assert inputs.aux.shape == (1, 10, 89, 90)
    # Everything is z-scored -> finite, roughly O(1) magnitude
    assert torch.isfinite(inputs.rho_cur).all()
    assert inputs.rho_cur.abs().mean() < 100.0


def test_forecast_inputs_rejects_launch_before_lag_window(demo_h5, demo_stats):
    drivers = frozen_drivers(kp=5.0, f107=165.0,
                             start="2025-11-10T00:00:00Z",
                             end="2025-11-13T00:00:00Z")
    # H5 starts at 2025-11-09T00:00Z; launch at 2025-11-10T12:00Z is 36h in,
    # which is < the model's 48h lag window.
    with pytest.raises(ValueError, match="lag window"):
        ForecastInputs.from_launch(
            h5_path=demo_h5,
            stats_path=demo_stats,
            launch_utc="2025-11-10T12:00:00Z",
            drivers=drivers,
        )


def test_aux_channels_match_lag_convention(demo_h5, demo_stats):
    """Aux tensor should have 10 channels in the documented order:
    [sin(lat), cos(lat), sin(lon), cos(lon), sin(doy), cos(doy),
     sin(hour), cos(hour), Kp, F10.7].

    Order verified against train_camnet_multilag.py:203-204 (driver[0]=Kp,
    driver[1]=F10.7) and Funs.py:482-483.
    """
    drivers = frozen_drivers(kp=8.7, f107=170.0,
                             start="2025-11-10T00:00:00Z",
                             end="2025-11-13T00:00:00Z")
    inputs = ForecastInputs.from_launch(
        h5_path=demo_h5,
        stats_path=demo_stats,
        launch_utc="2025-11-11T00:00:00Z",
        drivers=drivers,
    )
    aux = inputs.aux[0]  # (10, H, W)
    # Kp in channel 8, F10.7 in channel 9 - constant across (H, W)
    assert torch.allclose(aux[8], torch.full_like(aux[8], 8.7))
    assert torch.allclose(aux[9], torch.full_like(aux[9], 170.0))


def test_lag_hours_matches_training():
    """LAG_HOURS is the ground-truth lag convention; must be (24, 48)."""
    assert LAG_HOURS == (24, 48)
