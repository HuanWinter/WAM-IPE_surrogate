# WAMCast v0.1.0 Release Checklist

This checklist covers the manual actions to publish v0.1.0 on GitHub + Zenodo. All package code is already in place; only external service setup + placeholders need finalizing.

## Prerequisites

- [ ] GitHub personal account has push access to `github.com/HuanWinter/wamcast` (create the repo if it doesn't exist yet)
- [ ] Both authors have ORCID iDs. If not, register at https://orcid.org.
- [ ] Zenodo account linked to GitHub (https://zenodo.org/account/settings/github/)

## 1. Fill in placeholders

### 1a. ORCIDs

Replace `PLACEHOLDER-HU` and `PLACEHOLDER-MCC` in these two files with the real 16-digit ORCID iDs (with dashes, e.g. `0000-0002-1825-0097`):

- `CITATION.cff` (2 occurrences)
- `.zenodo.json` (2 occurrences)

Example: replace `"https://orcid.org/PLACEHOLDER-HU"` with `"https://orcid.org/0000-0002-1234-5678"`.

Verify: `grep -c PLACEHOLDER CITATION.cff .zenodo.json` should return `0`.

### 1b. Benjamin McCrossan's email

The `pyproject.toml` uses `mccrossan@hyperios.com` as a placeholder. Replace with the real address if different.

## 2. Push to GitHub

```bash
cd /media/faraday/andong/Workspace/wamcast

# Enable Zenodo integration BEFORE the first tag:
# https://zenodo.org/account/settings/github/ → toggle HuanWinter/wamcast ON
# (This is essential — Zenodo only mints DOIs for repos it was watching before the tag.)

# Create the repo on GitHub (skip if it already exists):
gh repo create HuanWinter/wamcast \
    --public \
    --description "Neural-operator WAM-IPE surrogate for storm-time thermospheric mass-density forecasting" \
    --source . \
    --push

# If gh isn't available / repo exists:
git remote add origin git@github.com:HuanWinter/wamcast.git  # or https://
git push -u origin main
```

## 3. Stage artifact bundle for Zenodo

The Zenodo bundle is ~30 GB. Stage it locally first:

```bash
mkdir -p /tmp/wamcast-release
cd /media/faraday/andong/Workspace/WAM-IPE

# 20 ensemble checkpoints (~28 GB total)
for i in $(seq -w 0 19); do
    cp Res/uq/ensemble_t16/member_${i}/best.ckpt \
       /tmp/wamcast-release/ensemble_t16_member_${i}.ckpt
done

# Stats + demo H5 + drivers
cp Res/ML_ready_stats_train_2025-06-30.npz /tmp/wamcast-release/train_stats.npz

# Test-period sub-H5 (need to build this — see note below)
# The full ML_ready_23-26_clean.h5 is ~30 GB. Create a subset containing only
# the 10 test storms + their 48h lag windows using scripts/extract_test_period_h5.py
# (script does not exist yet — write one or upload the full H5 if space allows).

# Test-period drivers
cp Res/ML_ready_23-26_clean_drivers.npz /tmp/wamcast-release/  # or export as CSV

# Cal-preds tarball (5 cal storms, ~500 MB)
tar czf /tmp/wamcast-release/cal_preds.tar.gz -C Res/uq/multihorizon ens_t16_storm{1,2,3,4,5}.npz

# Pre-fit calibrator
cp Res/uq/tables/calibrator_alpha05.npz /tmp/wamcast-release/calibrator_alpha05.npz

# List of test-storm launch UTCs
python -c "
import numpy as np, h5py, pandas as pd
cat = np.load('Res/uq/catalog_b_r16.npz', allow_pickle=True)
test = [(int(cat['storm_id'][i]), int(cat['start_idx'][i]))
        for i in range(len(cat['storm_id']))
        if str(cat['partition'][i]) == 'test']
with h5py.File('Res/ML_ready_23-26_clean.h5','r') as f:
    times = f['time'][:]
with open('/tmp/wamcast-release/launches_2025_test.txt','w') as g:
    for sid, si in test:
        g.write(pd.Timestamp(int(times[si]), tz='UTC').isoformat()+'\n')
"

# Verify size
du -sh /tmp/wamcast-release
```

## 4. Upload to Zenodo

Zenodo has a 50 GB per-record cap, but the web UI struggles with big files. Options:

**A. Web UI** (works for ≤ ~1 GB files)
1. https://zenodo.org/uploads/new — create a new upload
2. Metadata will be auto-populated from `.zenodo.json` on the next tag push, but for this manual upload, fill in:
   - Title, description, creators, keywords per `.zenodo.json`
   - Upload type: Software
   - License: MIT
3. Drag-drop the files from `/tmp/wamcast-release/`
4. Publish → note the DOI (format `10.5281/zenodo.XXXXXXX`)

**B. Zenodo REST API** (better for the 1.4 GB ckpts)

```bash
# One-time: get an access token from https://zenodo.org/account/settings/applications/tokens/new/
export ZENODO_TOKEN=<your token>

# Create a new deposition
curl -X POST https://zenodo.org/api/deposit/depositions \
    -H "Authorization: Bearer $ZENODO_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{}' | tee /tmp/deposition.json

DEPOSITION_ID=$(jq -r .id /tmp/deposition.json)
BUCKET_URL=$(jq -r .links.bucket /tmp/deposition.json)

# Upload each file
for f in /tmp/wamcast-release/*; do
    fname=$(basename "$f")
    echo "uploading $fname ..."
    curl -X PUT "$BUCKET_URL/$fname" \
        -H "Authorization: Bearer $ZENODO_TOKEN" \
        --upload-file "$f"
done

# Add metadata (paste contents of .zenodo.json)
curl -X PUT https://zenodo.org/api/deposit/depositions/$DEPOSITION_ID \
    -H "Authorization: Bearer $ZENODO_TOKEN" \
    -H "Content-Type: application/json" \
    -d "$(cat /media/faraday/andong/Workspace/wamcast/.zenodo.json | jq '{metadata: .}')"

# Publish
curl -X POST https://zenodo.org/api/deposit/depositions/$DEPOSITION_ID/actions/publish \
    -H "Authorization: Bearer $ZENODO_TOKEN"
```

Note the returned DOI (e.g. `10.5281/zenodo.12345678`) — you need the trailing number for Step 5.

## 5. Populate the ARTIFACTS registry

```bash
cd /media/faraday/andong/Workspace/wamcast
python scripts/emit_zenodo_manifest.py \
    --artifacts-dir /tmp/wamcast-release \
    --zenodo-record <REAL_RECORD_NUMBER> \
    > /tmp/artifacts_snippet.py

# Open /tmp/artifacts_snippet.py in an editor and copy its contents.
# Then edit src/wamcast/artifacts.py:
#   - Replace the _ZENODO_DOI and _ZENODO_BASE placeholder assignments
#   - Replace the empty ARTIFACTS = {} with the generated dict
```

## 6. Replace DOI placeholders in docs

```bash
cd /media/faraday/andong/Workspace/wamcast
RECORD=<REAL_RECORD_NUMBER>
sed -i "s|10.5281/zenodo.PLACEHOLDER|10.5281/zenodo.$RECORD|g" \
    README.md docs/reproducibility.md docs/zenodo.md src/wamcast/artifacts.py

grep -rn "PLACEHOLDER" README.md docs/ src/wamcast/ .zenodo.json CITATION.cff
# Expected: zero output.
```

## 7. Update CITATION.cff with the Zenodo DOI

Add under the `authors:` block:

```yaml
identifiers:
  - type: doi
    value: "10.5281/zenodo.<REAL_RECORD>"
    description: "Zenodo archive of code + trained checkpoints"
```

## 8. Verify + commit + tag + push

```bash
cd /media/faraday/andong/Workspace/wamcast

# Sanity: fast tests still pass
pytest -m "not slow and not gpu" -q

# Sanity: download-artifacts now works
wamcast download-artifacts --name train_stats --cache-dir /tmp/wamcast-cache
ls -la /tmp/wamcast-cache/train_stats.npz  # should exist, hash-verified

# Commit + tag + push
git add -u
git commit -m "release: v0.1.0 (DOI 10.5281/zenodo.<REAL_RECORD>)"
git tag -a v0.1.0 -m "WAMCast v0.1.0 — inference-only companion to Hu & McCrossan (2026)"
git push origin main --tags
```

The tag push triggers `.github/workflows/release.yml`, which builds sdist+wheel and attaches them to the GitHub Release. Zenodo's GitHub webhook mints a new version DOI on the source archive (separate from the artifact-bundle DOI above; both link back to the release).

## 9. Wire the DOI into the manuscript (Task 16)

In `/media/faraday/andong/Workspace/WAM-IPE/Paper/manuscript_chronosplit.tex` and its Overleaf mirror at `Paper/wam_ipe_paper/chronosplit/manuscript_chronosplit.tex`, find the Software availability paragraph and replace:

```
A frozen release of the code, the 20-seed model checkpoints ($T{=}16$ and $T{=}4$), and the sanitised WFS baseline archive will be minted on Zenodo at acceptance (DOI to follow).
```

with:

```
An inference-only Python package, WAMCast, is available at \url{https://github.com/HuanWinter/wamcast} (v0.1.0) with 20-member T=16 checkpoints, the pre-fit split-conformal calibrator, and demo ML-ready HDF5 subsets covering the 10 test storms archived on Zenodo at \url{https://doi.org/10.5281/zenodo.<REAL_RECORD>}. Reproducing every headline number in Table~\ref{tab:headline_per_horizon} from the archived artifacts is documented in \texttt{docs/reproducibility.md}.
```

Then commit and push to Overleaf (the main-tree GitHub push remains blocked by the .npz history issue).

## What Task 15 does NOT do

- Fill in real ORCID iDs (author must provide)
- Retire the torch>=2.6 weights_only shim (see `wamcast.model.load_wamcast_from_checkpoint`) — that helper stays as a permanent compatibility layer; the alternative is re-minting all 20 ckpts under a torch<2.6 or newer-safeguarded pickle protocol.
- Push the main WAM-IPE research repo to GitHub (the ~2.5 GB .npz blob history remains blocked — a separate `git filter-repo` + force-push decision).
