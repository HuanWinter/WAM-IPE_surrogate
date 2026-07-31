"""Fit a conformal calibrator on cal storms, apply to a test forecast.

Usage:
    python calibrate_and_predict.py \\
        --cal-preds cal_storm{1..5}.npz \\
        --test-pred test_storm11.npz \\
        --alpha 0.05
"""
from __future__ import annotations

import argparse

import numpy as np

from wamcast.conformal import Calibrator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal-preds", nargs="+", required=True,
                    help="Cal-storm prediction NPZs (mu/sigma/truth/meta).")
    ap.add_argument("--test-pred", required=True,
                    help="Test-storm prediction NPZ.")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    cal = []
    for p in args.cal_preds:
        d = np.load(p, allow_pickle=True)
        cal.append(dict(mu=d["mu"], sigma=d["sigma"], truth=d["truth"],
                        meta=d["meta"].item()))
    c = Calibrator.fit(cal, alpha=args.alpha)

    t = np.load(args.test_pred, allow_pickle=True)
    lo, hi = c.intervals(dict(mu=t["mu"], sigma=t["sigma"],
                              meta=t["meta"].item()))
    coverage = float(((t["truth"] >= lo) & (t["truth"] <= hi)).mean())
    width = float((hi - lo).mean())
    print(f"alpha={args.alpha}  coverage={coverage:.3f}  mean_width={width:.3f}")


if __name__ == "__main__":
    main()
