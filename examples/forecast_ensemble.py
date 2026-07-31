"""20-member ensemble forecast with observed OMNI drivers.

Usage:
    python forecast_ensemble.py --ens-dir path/to/ensemble_t16 --h5 path/to/ml_ready.h5 \\
        --stats path/to/stats.npz --omni-csv path/to/omni2.csv --launch 2025-11-11T00:00:00Z
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wamcast.dataset import ForecastInputs
from wamcast.drivers import load_omni_csv
from wamcast.model import load_wamcast_from_checkpoint
from wamcast.rollout import ensemble_rollout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens-dir", required=True,
                    help="Dir with member_00/.../member_19/best.ckpt")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--omni-csv", required=True)
    ap.add_argument("--launch", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ens = Path(args.ens_dir)
    members = [
        load_wamcast_from_checkpoint(
            str(ens / f"member_{i:02d}" / "best.ckpt"), map_location=args.device,
        ).to(args.device).eval()
        for i in range(20)
    ]
    drivers = load_omni_csv(args.omni_csv, start=args.launch,
                            end=f"{args.launch[:10]}T23:59:59Z")
    inputs = ForecastInputs.from_launch(
        h5_path=args.h5, stats_path=args.stats,
        launch_utc=args.launch, drivers=drivers,
    )
    fcst = ensemble_rollout(members, inputs, drivers=drivers,
                            h5_path=args.h5, stats_path=args.stats)
    for h, mu, sig in zip(fcst.horizons_hours, fcst.mu, fcst.sigma, strict=True):
        print(f"+{h:>2}h: |mu|={mu.abs().mean():.3f}  |sigma|={sig.mean():.3f}")


if __name__ == "__main__":
    main()
