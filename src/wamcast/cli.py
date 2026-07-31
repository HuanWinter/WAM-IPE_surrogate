"""Click-based CLI: `wamcast forecast | calibrate | download-artifacts`."""
from __future__ import annotations

from pathlib import Path

import click
import h5py
import numpy as np

from wamcast import __version__
from wamcast.artifacts import ARTIFACTS, download
from wamcast.conformal import Calibrator
from wamcast.dataset import ForecastInputs, LEV_HI, LEV_LO
from wamcast.drivers import frozen_drivers, load_omni_csv, load_swpc_rsga_json
from wamcast.io import ForecastMetadata, write_forecast_netcdf
from wamcast.model import WAMCastModel
from wamcast.rollout import ensemble_rollout, rollout


@click.group()
@click.version_option(__version__)
def main() -> None:
    """WAMCast — neural-operator WAM-IPE surrogate forecasts."""


@main.command("forecast")
@click.option("--ckpt", "ckpts", multiple=True, required=True, type=click.Path(exists=True),
              help="Path to a checkpoint (repeat for ensemble).")
@click.option("--h5", "h5_path", required=True, type=click.Path(exists=True),
              help="ML-ready H5 with launch frame + prior 48h lag frames.")
@click.option("--stats", "stats_path", required=True, type=click.Path(exists=True),
              help="Per-level mean/std NPZ (train-only).")
@click.option("--launch", required=True, help="Launch UTC (ISO 8601).")
@click.option("--driver-protocol",
              type=click.Choice(["observed_omni", "swpc_forecast", "frozen"]),
              default="frozen")
@click.option("--omni-csv", type=click.Path(exists=True))
@click.option("--swpc-json", type=click.Path(exists=True))
@click.option("--frozen-kp", type=float)
@click.option("--frozen-f107", type=float)
@click.option("--horizons", default="3,6,12,24,48",
              help="Comma-separated hours.")
@click.option("--calibrator", "calibrator_path", type=click.Path(exists=True),
              help="Optional split-conformal calibrator NPZ (from `wamcast calibrate`).")
@click.option("--out", required=True, type=click.Path())
@click.option("--device", default="cuda", type=click.Choice(["cpu", "cuda"]))
def cli_forecast(ckpts, h5_path, stats_path, launch, driver_protocol,
                 omni_csv, swpc_json, frozen_kp, frozen_f107,
                 horizons, calibrator_path, out, device):
    """Run a WAMCast forecast for one launch time."""
    horizons_tuple = tuple(int(h) for h in horizons.split(","))
    # Drivers cover launch + longest horizon (+48h buffer)
    import pandas as pd
    launch_ts = pd.Timestamp(launch)
    if launch_ts.tz is None:
        launch_ts = launch_ts.tz_localize("UTC")
    end_ts = launch_ts + pd.Timedelta(hours=max(horizons_tuple) + 24)
    end_iso = end_ts.isoformat()

    if driver_protocol == "observed_omni":
        if omni_csv is None:
            raise click.UsageError("--omni-csv required for observed_omni protocol")
        drivers = load_omni_csv(omni_csv, start=launch, end=end_iso)
    elif driver_protocol == "swpc_forecast":
        if swpc_json is None:
            raise click.UsageError("--swpc-json required for swpc_forecast protocol")
        drivers = load_swpc_rsga_json(swpc_json, start=launch, end=end_iso)
    else:  # frozen
        if frozen_kp is None or frozen_f107 is None:
            raise click.UsageError("--frozen-kp and --frozen-f107 required for frozen protocol")
        drivers = frozen_drivers(kp=frozen_kp, f107=frozen_f107,
                                 start=launch, end=end_iso)

    inp = ForecastInputs.from_launch(
        h5_path=h5_path, stats_path=stats_path,
        launch_utc=launch, drivers=drivers,
    )
    members = [WAMCastModel.load_from_checkpoint(c, map_location=device).eval()
               for c in ckpts]
    for m in members:
        m.to(device)

    if len(members) == 1:
        fcst = rollout(members[0], inp, drivers=drivers,
                       h5_path=h5_path, stats_path=stats_path,
                       horizons_hours=horizons_tuple)
        variant = "standard_t16_single"
    else:
        fcst = ensemble_rollout(members, inp, drivers=drivers,
                                h5_path=h5_path, stats_path=stats_path,
                                horizons_hours=horizons_tuple)
        variant = f"standard_t16_ensemble_k{len(members)}"

    with h5py.File(h5_path, "r") as f:
        lat = f["lat"][1:-1].astype(np.float32)
        lon = f["lon"][:].astype(np.float32)

    lo = hi = None
    alpha = None
    if calibrator_path is not None:
        cal = Calibrator.load(calibrator_path)
        peak_kp = float(max(drivers.kp))
        lo_stack, hi_stack = [], []
        for h_idx in range(fcst.mu.shape[0]):
            lo_h, hi_h = cal.intervals(dict(
                mu=fcst.mu[h_idx].numpy(),
                sigma=fcst.sigma[h_idx].numpy(),
                meta=dict(peak_kp=peak_kp),
            ))
            lo_stack.append(lo_h)
            hi_stack.append(hi_h)
        lo = np.stack(lo_stack, axis=0)
        hi = np.stack(hi_stack, axis=0)
        alpha = cal.alpha

    meta = ForecastMetadata(
        launch_utc=inp.launch_utc,
        driver_protocol=driver_protocol,
        model_variant=variant,
        checkpoint_paths=tuple(str(c) for c in ckpts),
        wamcast_version=__version__,
    )
    write_forecast_netcdf(
        out,
        mu=fcst.mu.numpy(), sigma=fcst.sigma.numpy(),
        horizons_hours=fcst.horizons_hours,
        lat=lat, lon=lon,
        level=np.arange(LEV_LO, LEV_HI, dtype=np.int32),
        metadata=meta,
        lo=lo, hi=hi, alpha=alpha,
    )
    click.echo(f"wrote {out}")


@main.command("calibrate")
@click.option("--cal-preds", multiple=True, required=True, type=click.Path(exists=True),
              help="Calibration prediction NPZs (repeat per storm).")
@click.option("--alpha", type=float, default=0.05)
@click.option("--out", required=True, type=click.Path())
def cli_calibrate(cal_preds, alpha, out):
    """Fit a split-conformal calibrator on cal-set predictions."""
    preds = []
    for p in cal_preds:
        d = np.load(p, allow_pickle=True)
        preds.append(dict(mu=d["mu"], sigma=d["sigma"], truth=d["truth"],
                          meta=d["meta"].item()))
    c = Calibrator.fit(preds, alpha=alpha)
    c.save(out)
    click.echo(f"fit calibrator (alpha={alpha}) on {len(preds)} storms -> {out}")


@main.command("download-artifacts")
@click.option("--name", multiple=True, help="Specific artifact name (repeat). Default: all.")
@click.option("--cache-dir", type=click.Path(), default=None)
def cli_download(name, cache_dir):
    """Download WAMCast Zenodo artifacts (checkpoints, demo data) to a cache."""
    cache = Path(cache_dir) if cache_dir else None
    names = list(name) if name else list(ARTIFACTS.keys())
    if not names:
        raise click.ClickException(
            "No artifacts registered - wait for the Zenodo mint (see docs/zenodo.md)."
        )
    for n in names:
        got = download(ARTIFACTS[n], cache_dir=cache)
        click.echo(str(got))
