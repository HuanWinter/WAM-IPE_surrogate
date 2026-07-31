"""Tests for OMNI2 / SWPC / frozen driver ingest."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from wamcast.drivers import (
    DriverSeries, load_omni_csv, load_swpc_rsga_json, frozen_drivers,
)

DATA = pathlib.Path(__file__).parent / "data"


def test_load_omni_csv_returns_kp_f107_on_10min_grid():
    ds = load_omni_csv(DATA / "demo_omni.csv",
                       start="2025-11-10T00:00:00Z",
                       end="2025-11-12T00:00:00Z")
    assert isinstance(ds, DriverSeries)
    # 48 hours * 6 samples/hour + 1 endpoint = 289
    assert len(ds.time) == 289
    assert ds.kp.shape == (289,)
    assert ds.f107.shape == (289,)
    # Storm peak on Nov 11 03Z should propagate through forward-fill
    peak_idx = int((pd.Timestamp("2025-11-11T03:00:00Z") -
                    pd.Timestamp("2025-11-10T00:00:00Z")).total_seconds() // 600)
    assert ds.kp[peak_idx] == pytest.approx(8.7)


def test_load_swpc_rsga_produces_daily_step_forecast():
    ds = load_swpc_rsga_json(DATA / "demo_swpc_rsga.json",
                             start="2025-11-11T00:00:00Z",
                             end="2025-11-13T00:00:00Z")
    # Kp is daily-max held constant across each UT day
    kp_at_nov11_12z = ds.kp[int(12 * 6)]
    kp_at_nov12_00z = ds.kp[int(24 * 6)]
    assert kp_at_nov11_12z == pytest.approx(7.0)
    assert kp_at_nov12_00z == pytest.approx(5.0)


def test_frozen_drivers_hold_scalars_across_window():
    ds = frozen_drivers(kp=6.7, f107=165.0,
                        start="2025-11-11T00:00:00Z",
                        end="2025-11-13T00:00:00Z")
    assert (ds.kp == 6.7).all()
    assert (ds.f107 == 165.0).all()


def test_driver_series_rejects_wrong_length_mismatch():
    """Constructing DriverSeries with mismatched-length arrays should error."""
    from wamcast.drivers import DriverSeries
    with pytest.raises(ValueError, match="length mismatch"):
        DriverSeries(
            time=pd.date_range("2025-11-11", periods=10, freq="10min"),
            kp=np.zeros(10),
            f107=np.zeros(11),
        )


def test_as_h5_driver_array_column_order_matches_trained_model():
    """Column order must be [Kp, F10.7] — the order the model was trained on.

    Regression guard for a bug that would silently swap Kp/F10.7 channels in
    the model's aux tensor with no crash and no test signal (just physically
    wrong forecasts). Evidence for the correct order:
    - WAM-IPE/scripts/backfill_drivers_2025.py:103 writes columns [Kp, F107]
    - WAM-IPE/train_camnet_multilag.py:203-204 reads driver[0]=Kp, driver[1]=F107
    - WAM-IPE/Funs.py:482-483 confirms same convention
    """
    ds = DriverSeries(
        time=pd.date_range("2025-11-11", periods=3, freq="10min", tz="UTC"),
        kp=np.array([5.0, 6.0, 7.0], dtype=np.float32),
        f107=np.array([100.0, 150.0, 200.0], dtype=np.float32),
    )
    arr = ds.as_h5_driver_array()
    assert arr.shape == (3, 2)
    # Column 0 must be Kp (5, 6, 7 range), not F10.7 (100, 150, 200 range)
    assert (arr[:, 0] < 10).all(), f"col 0 should be Kp but got {arr[:, 0]}"
    assert (arr[:, 1] > 50).all(), f"col 1 should be F10.7 but got {arr[:, 1]}"


def test_load_swpc_rsga_rejects_zero_overlap_window():
    """Requesting a window that doesn't overlap the bulletin should error, not
    silently return an interpolated/NaN-filled series."""
    with pytest.raises(ValueError, match="zero overlap"):
        load_swpc_rsga_json(DATA / "demo_swpc_rsga.json",
                            start="2030-01-01T00:00:00Z",
                            end="2030-01-02T00:00:00Z")


def test_grid_normalizes_mixed_tz_inputs():
    """Mixed tz-aware / naive inputs should both be normalized to UTC."""
    ds1 = frozen_drivers(kp=5.0, f107=100.0,
                         start="2025-11-11T00:00:00",       # naive
                         end="2025-11-11T01:00:00Z")        # aware
    ds2 = frozen_drivers(kp=5.0, f107=100.0,
                         start="2025-11-11T00:00:00Z",
                         end="2025-11-11T01:00:00Z")
    assert len(ds1.time) == len(ds2.time)
    assert (ds1.time == ds2.time).all()
