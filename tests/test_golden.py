"""Golden test: reproduce a paper headline number from packaged inference.

Marked `slow` and `gpu`. Skipped in fast CI; run manually on a GPU box:

    pytest -m 'slow and gpu' tests/test_golden.py

If the shim in wamcast.model.load_wamcast_from_checkpoint is insufficient
for the ensemble checkpoints, this test fails at load time with the same
numpy globals error as test_model.py's xfail. The fix is to extend the
allowlist in that helper.
"""
from __future__ import annotations

import pathlib

import pytest

pytest.importorskip(
    "torch_harmonics",
    reason="torch_harmonics C extension unavailable (typical on CI without a matching torch ABI)",
)

import numpy as np  # noqa: E402
import torch  # noqa: E402

RESEARCH = pathlib.Path("/media/faraday/andong/Workspace/WAM-IPE")
H5 = RESEARCH / "Res" / "ML_ready_23-26_clean.h5"
STATS = RESEARCH / "Res" / "ML_ready_stats_train_2025-06-30.npz"
SIDECAR = RESEARCH / "Res" / "ML_ready_23-26_clean_drivers.npz"
ENS_DIR = RESEARCH / "Res" / "uq" / "ensemble_t16"
CATALOG = RESEARCH / "Res" / "uq" / "catalog_b_r16.npz"

# Storm 11 = 2025-11-11 super-storm. Expected +48h MAE (z-score) is looked
# up at test runtime from the manuscript's own headline production file:
# Res/uq/multihorizon/ens_t16_storm11.npz meta.per_horizon_mae[48].
STORM_ID = 11
TOL_REL = 0.02  # 2% relative — accommodates cudnn nondeterminism seed-to-seed


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(not H5.exists(), reason="research H5 not present")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_reproduce_storm11_48h_mae_from_paper():
    """Reproduce paper's storm-11 +48h grid MAE end-to-end.

    Loads the manuscript's ensemble ckpts (20 members) and paper's own
    reference NPZ, runs packaged inference through the same launch time,
    asserts +48h MAE matches to TOL_REL.
    """
    if not ENS_DIR.exists():
        pytest.skip("ensemble ckpts not present")

    # Load paper's own +48h MAE from the headline production NPZ.
    ref_npz = RESEARCH / "Res" / "uq" / "multihorizon" / f"ens_t16_storm{STORM_ID}.npz"
    if not ref_npz.exists():
        pytest.skip(f"reference NPZ {ref_npz} not present")
    ref = np.load(ref_npz, allow_pickle=True)
    ref_meta = ref["meta"].item()
    expected_mae_48h = float(ref_meta["per_horizon_mae"][48])

    # Load launch time from catalog.
    import h5py
    import pandas as pd
    cat = np.load(CATALOG, allow_pickle=True)
    idx_in_cat = int(np.where(cat["storm_id"] == STORM_ID)[0][0])
    with h5py.File(H5, "r") as f:
        launch_ns = int(f["time"][int(cat["start_idx"][idx_in_cat])])
    launch_utc = pd.Timestamp(launch_ns, tz="UTC").isoformat()

    # Observed OMNI drivers from the training sidecar. Sidecar column order:
    # [Kp, F10.7] — see WAM-IPE/scripts/backfill_drivers_2025.py:103.
    sc = np.load(SIDECAR)
    with h5py.File(H5, "r") as f:
        times = pd.to_datetime(f["time"][:], utc=True)
    from wamcast.drivers import DriverSeries
    drivers = DriverSeries(
        time=times, kp=sc["driver"][:, 0], f107=sc["driver"][:, 1],
    )

    from wamcast.dataset import ForecastInputs
    from wamcast.model import load_wamcast_from_checkpoint
    from wamcast.rollout import ensemble_rollout

    inp = ForecastInputs.from_launch(
        h5_path=H5, stats_path=STATS,
        launch_utc=launch_utc, drivers=drivers,
    )
    members = [load_wamcast_from_checkpoint(
                   str(ENS_DIR / f"member_{i:02d}" / "best.ckpt"),
                   map_location="cuda").cuda().eval()
               for i in range(20)]
    fcst = ensemble_rollout(members, inp, drivers=drivers,
                            h5_path=H5, stats_path=STATS)

    # +48h is the last horizon in DEFAULT_HORIZONS_HOURS = (3, 6, 12, 24, 48)
    mu_48 = fcst.mu[-1, 0].cpu().numpy()  # (C, lat, lon), z-scored

    with h5py.File(H5, "r") as f:
        launch_frame = int(np.searchsorted(f["time"][:], launch_ns))
        truth_raw = f["rho"][launch_frame + 16 * 18, 10:51, 1:-1, :]
    truth_raw = np.nan_to_num(truth_raw, nan=0.0)
    mean = np.load(STATS)["mean"][10:51][:, None, None]
    std = np.load(STATS)["std"][10:51][:, None, None]
    truth_z = ((truth_raw - mean) / std).astype(np.float32)
    got_mae = float(np.abs(mu_48 - truth_z).mean())

    rel = abs(got_mae - expected_mae_48h) / expected_mae_48h
    assert rel < TOL_REL, (
        f"packaged inference drifts from paper: "
        f"got={got_mae:.5f}, expected={expected_mae_48h:.5f}, "
        f"|Δ|/expected={rel:.3%}"
    )
