"""Forecast one launch with a single WAMCast checkpoint.

Usage:
    python forecast_single_launch.py \\
        --ckpt path/to/best.ckpt \\
        --h5 path/to/ml_ready.h5 \\
        --stats path/to/stats.npz \\
        --launch 2025-11-11T00:00:00Z \\
        --kp 6.7 --f107 168.0
"""
from __future__ import annotations

import argparse

from wamcast.dataset import ForecastInputs
from wamcast.drivers import frozen_drivers
from wamcast.model import load_wamcast_from_checkpoint
from wamcast.rollout import rollout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--launch", required=True, help="UTC ISO 8601")
    ap.add_argument("--kp", type=float, default=6.7)
    ap.add_argument("--f107", type=float, default=168.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    drivers = frozen_drivers(
        kp=args.kp, f107=args.f107,
        start=args.launch, end=f"{args.launch[:10]}T23:59:59Z",
    )
    inputs = ForecastInputs.from_launch(
        h5_path=args.h5, stats_path=args.stats,
        launch_utc=args.launch, drivers=drivers,
    )
    model = load_wamcast_from_checkpoint(args.ckpt, map_location=args.device)
    model.to(args.device).eval()
    fcst = rollout(model, inputs, drivers=drivers,
                   h5_path=args.h5, stats_path=args.stats)
    for h, mu in zip(fcst.horizons_hours, fcst.mu, strict=True):
        print(f"+{h:>2}h: mu shape={tuple(mu.shape)}  |mu|={mu.abs().mean():.3f}")


if __name__ == "__main__":
    main()
