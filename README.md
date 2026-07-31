# WAMCast

Neural-operator WAM-IPE surrogate for storm-time thermospheric mass-density forecasting with calibrated uncertainty. Inference-only companion to Hu & McCrossan (submitted, JGR:MLC).

## Install

```bash
pip install wamcast
```

Or from source:

```bash
git clone https://github.com/HuanWinter/wamcast.git
cd wamcast
pip install -e ".[test,dev]"
```

## Quick forecast

```bash
wamcast download-artifacts --name ensemble_t16_member_00 --name demo_launch_h5
wamcast forecast \
    --ckpt ~/.cache/wamcast/ensemble_t16_member_00.ckpt \
    --h5 ~/.cache/wamcast/demo_launch.h5 \
    --stats ~/.cache/wamcast/train_stats.npz \
    --launch 2025-11-11T00:00:00Z \
    --driver-protocol frozen --frozen-kp 8.7 --frozen-f107 170 \
    --out forecast.nc
```

Output: CF-1.10 NetCDF with `rho_mu`, `rho_sigma`, and (with `--calibrator`) `rho_lo`/`rho_hi` at the requested horizons.

## Documentation

- [Quickstart](docs/quickstart.md) — install → download → first forecast → read with xarray
- [Input formats](docs/input-formats.md) — H5 schema, OMNI2 CSV, SWPC RSGA JSON
- [Output format](docs/output-format.md) — NetCDF variables, coordinates
- [Reproducing the paper](docs/reproducibility.md) — step-by-step from the Zenodo DOI
- [CLI reference](docs/cli.md)
- [Zenodo release](docs/zenodo.md) — what's on the DOI vs what's in git

## Citation

```bibtex
@article{hu2026wamcast,
  title   = {WAMCast: A Neural-Operator WAM-IPE Surrogate Model for Storm-Time
             Thermospheric Mass-Density Forecasting with Calibrated Uncertainty},
  author  = {Hu, Andong and McCrossan, Benjamin},
  journal = {Journal of Geophysical Research: Machine Learning and Computation},
  year    = {2026},
  note    = {In review}
}
```

Frozen release (checkpoints, demo H5, calibrator NPZ) archived on Zenodo: [DOI:10.5281/zenodo.PLACEHOLDER](https://doi.org/10.5281/zenodo.PLACEHOLDER).

## License

MIT — see [LICENSE](LICENSE).
