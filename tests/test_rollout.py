"""Tests for rollout.rollout() and rollout.ensemble_rollout()."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from wamcast.dataset import ForecastInputs
from wamcast.drivers import frozen_drivers
from wamcast.model import WAMCastModel
from wamcast.rollout import ensemble_rollout, rollout


@pytest.fixture
def tiny_model():
    """A 1-block SFNO for fast CPU tests. Untrained → outputs are junk but
    shape-correct."""
    return WAMCastModel(n_blocks=1, hidden_ch=32, modes_l=4, modes_m=4).eval()


@pytest.fixture
def inputs(demo_h5, demo_stats):
    drivers = frozen_drivers(kp=6.7, f107=168.0,
                             start="2025-11-09T00:00:00Z",
                             end="2025-11-14T00:00:00Z")
    return ForecastInputs.from_launch(
        h5_path=demo_h5, stats_path=demo_stats,
        launch_utc="2025-11-11T00:00:00Z", drivers=drivers,
    ), drivers, demo_h5, demo_stats


def test_rollout_default_horizons(tiny_model, inputs):
    inp, drivers, h5, stats = inputs
    out = rollout(tiny_model, inp, drivers=drivers, h5_path=h5, stats_path=stats)
    assert set(out.horizons_hours) == {3, 6, 12, 24, 48}
    assert out.mu.shape == (5, 1, 41, 89, 90)   # (H, B, C, lat, lon)
    assert torch.isfinite(out.mu).all()


def test_rollout_custom_horizons(tiny_model, inputs):
    inp, drivers, h5, stats = inputs
    out = rollout(tiny_model, inp, horizons_hours=(3, 12, 48),
                  drivers=drivers, h5_path=h5, stats_path=stats)
    assert out.horizons_hours == (3, 12, 48)
    assert out.mu.shape == (3, 1, 41, 89, 90)


def test_ensemble_rollout_produces_mu_and_sigma(inputs):
    inp, drivers, h5, stats = inputs
    members = [WAMCastModel(n_blocks=1, hidden_ch=32, modes_l=4, modes_m=4).eval()
               for _ in range(3)]
    out = ensemble_rollout(members, inp, drivers=drivers,
                           h5_path=h5, stats_path=stats)
    assert out.mu.shape == (5, 1, 41, 89, 90)
    assert out.sigma.shape == (5, 1, 41, 89, 90)
    assert (out.sigma > 0).all()
