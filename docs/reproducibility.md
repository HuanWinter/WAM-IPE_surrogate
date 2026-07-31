# Reproducing the paper headline numbers

This walks through reproducing the +48h grid MAE and satellite RMSE numbers reported in Hu & McCrossan (2026), Table 1 and Table 2, using only the WAMCast package and the Zenodo release.

## 1. Install

```bash
python -m pip install wamcast
```

## 2. Download artifacts

```bash
wamcast download-artifacts
```

This fetches from Zenodo (DOI: `10.5281/zenodo.PLACEHOLDER`):
- `ensemble_t16_member_00.ckpt` ... `member_19.ckpt` — 20 × ~1.4 GB
- `train_stats.npz` — per-level mean/std (train-only cutoff 2025-07-01)
- `ml_ready_2025.h5` — ML-ready HDF5 with the 10 test storms + their 48h lag windows (~4.2 GB)
- `omni2_test_period.csv` — observed OMNI2 Kp/F10.7 over the test window
- `swpc_rsga_test_period.json` — matching SWPC daily bulletins
- `cal_preds.tar` — cal-storm prediction NPZs (for fitting the calibrator yourself)
- `calibrator_alpha05.npz` — pre-fit calibrator (skip Step 4 if using this)

## 3. Run all 10 test storms

The Zenodo bundle includes `launches_2025_test.txt` — one launch UTC per line:

```bash
mkdir -p out
while read launch; do
    wamcast forecast \
        --ckpt ~/.cache/wamcast/ensemble_t16_member_*.ckpt \
        --h5 ~/.cache/wamcast/ml_ready_2025.h5 \
        --stats ~/.cache/wamcast/train_stats.npz \
        --launch "$launch" \
        --driver-protocol observed_omni \
        --omni-csv ~/.cache/wamcast/omni2_test_period.csv \
        --calibrator ~/.cache/wamcast/calibrator_alpha05.npz \
        --out out/forecast_${launch}.nc
done < ~/.cache/wamcast/launches_2025_test.txt
```

## 4. Fit your own calibrator (optional)

```bash
wamcast calibrate \
    --cal-preds ~/.cache/wamcast/cal_preds/*.npz \
    --alpha 0.05 \
    --out my_calibrator.npz
```

## Driver protocol variants

Three protocols reported in the paper:

| Protocol                          | CLI flag                                                     | Paper column      |
|-----------------------------------|--------------------------------------------------------------|-------------------|
| Observed OMNI2 (leaky upper bound) | `--driver-protocol observed_omni --omni-csv …`               | "observed"        |
| SWPC forecast (operational anchor) | `--driver-protocol swpc_forecast --swpc-json …`              | "SWPC-forecast"   |
| Frozen at launch (ablation floor)  | `--driver-protocol frozen --frozen-kp K --frozen-f107 F`     | "frozen"          |

## Golden test

The repo includes `tests/test_golden.py` which reproduces storm-11's +48h MAE (paper's headline value from `Res/uq/multihorizon/ens_t16_storm11.npz`) to within 2% relative. Run manually:

```bash
pytest -m 'slow and gpu' tests/test_golden.py -v
```

Currently: PASSES on the author's GPU box if the safe-globals shim covers all numpy globals in the ckpts. If it fails with a `_pickle.UnpicklingError`, extend the allowlist in `src/wamcast/model.py:load_wamcast_from_checkpoint`.
