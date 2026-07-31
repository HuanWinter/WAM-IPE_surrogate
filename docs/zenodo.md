# Zenodo release

The frozen artifacts required to reproduce the manuscript's headline numbers live on Zenodo, not in git. GitHub carries only source code + tests + docs (< 100 MB); Zenodo carries checkpoints + demo H5 + reference NPZs (~30 GB).

## What's on Zenodo (DOI: 10.5281/zenodo.PLACEHOLDER)

| Artifact                              | Size    | Purpose |
|---------------------------------------|---------|---------|
| `ensemble_t16_member_00..19.ckpt`     | ~28 GB  | 20-member T=16 grid ensemble (paper's headline grid production) |
| `train_stats.npz`                     | few KB  | Per-level mean/std, train-only cutoff 2025-07-01 |
| `ml_ready_2025.h5`                    | ~4 GB   | Subset of the training H5 with the 10 test storms + 48h lag windows |
| `omni2_test_period.csv`               | few KB  | Observed OMNI2 Kp/F10.7 for the 2025 test window |
| `swpc_rsga_test_period.json`          | few KB  | Matching SWPC daily bulletins |
| `cal_preds.tar.gz`                    | ~500 MB | Cal-storm prediction NPZs (for fitting the calibrator yourself) |
| `calibrator_alpha05.npz`              | few KB  | Pre-fit calibrator at α=0.05 |

## Populating the ARTIFACTS registry

At release time, `scripts/emit_zenodo_manifest.py` (planned) computes SHA-256 + size for every uploaded artifact and emits the `ARTIFACTS` dict in `src/wamcast/artifacts.py`. Until Task 15 is executed, the registry is empty and `wamcast download-artifacts` fails with a "no artifacts registered" message.

## Hash verification

`wamcast.artifacts.download()` verifies SHA-256 against the registry before writing to `~/.cache/wamcast/`. A mismatch raises `ValueError` and deletes the partial file. Cached files are re-verified on every call, so a re-download only happens if the cache is corrupted or absent.

## GitHub-Zenodo integration

The release workflow at `.github/workflows/release.yml` triggers on `v*` tag push and builds an sdist+wheel. When Zenodo's GitHub integration is enabled for the repo, the same tag push mints a fresh Zenodo DOI on the source archive. Uploading the ~30 GB artifacts to that Zenodo record is a separate manual step (Zenodo caps individual API uploads and the ckpt bundle is unwieldy over HTTP).
