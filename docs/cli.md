# CLI reference

## `wamcast forecast`

```
Usage: wamcast forecast [OPTIONS]

  Run a WAMCast forecast for one launch time.

Options:
  --ckpt PATH               Path to a checkpoint (repeat for ensemble).  [required]
  --h5 PATH                 ML-ready H5 with launch frame + 48h lag frames.  [required]
  --stats PATH              Per-level mean/std NPZ.  [required]
  --launch TEXT             Launch UTC (ISO 8601).  [required]
  --driver-protocol         observed_omni | swpc_forecast | frozen  [default: frozen]
  --omni-csv PATH           Required for observed_omni protocol.
  --swpc-json PATH          Required for swpc_forecast protocol.
  --frozen-kp FLOAT         Required for frozen protocol.
  --frozen-f107 FLOAT       Required for frozen protocol.
  --horizons TEXT           Comma-separated hours.  [default: 3,6,12,24,48]
  --calibrator PATH         Optional split-conformal calibrator NPZ.
  --out PATH                [required]
  --device                  cpu | cuda  [default: cuda]
```

## `wamcast calibrate`

```
Usage: wamcast calibrate [OPTIONS]

  Fit a split-conformal calibrator on cal-set predictions.

Options:
  --cal-preds PATH  Calibration prediction NPZs (repeat per storm).  [required]
  --alpha FLOAT     [default: 0.05]
  --out PATH        [required]
```

## `wamcast download-artifacts`

```
Usage: wamcast download-artifacts [OPTIONS]

  Download WAMCast Zenodo artifacts (checkpoints, demo data) to a cache.

Options:
  --name TEXT       Specific artifact name (repeat). Default: all.
  --cache-dir PATH  Defaults to ~/.cache/wamcast/.
```
