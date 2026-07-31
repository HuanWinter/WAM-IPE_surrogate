# Output format

WAMCast writes a CF-1.10 compliant NetCDF file with the following schema:

## Coordinates

| Coordinate  | Dtype   | Units          | Notes |
|-------------|---------|----------------|-------|
| `horizon`   | int32   | `h`            | Forecast horizon in hours since launch (`long_name` = "forecast horizon (hours since launch)") |
| `level`     | int32   | —              | WAM-IPE pressure level index (10..50) |
| `latitude`  | float32 | `degrees_north` | -88..88 (89 interior rows) |
| `longitude` | float32 | `degrees_east`  | 0..358 (90 columns) |

## Data variables

All variables are z-scored (subtract per-level mean, divide by per-level std from the training statistics NPZ). Users converting back to physical density should apply the inverse using the same stats file.

| Variable    | Shape                                      | Units    | Notes |
|-------------|---------------------------------------------|----------|-------|
| `rho_mu`    | `(horizon, level, latitude, longitude)`    | z-score  | Predicted mean |
| `rho_sigma` | `(horizon, level, latitude, longitude)`    | z-score  | Predicted std (0 for single-member, ensemble std for K-member) |
| `rho_lo`    | (same)                                     | z-score  | Lower conformal bound (present only if `--calibrator` was passed) |
| `rho_hi`    | (same)                                     | z-score  | Upper conformal bound |

## Global attributes

| Attribute            | Example                                     |
|----------------------|---------------------------------------------|
| `Conventions`        | `CF-1.10`                                    |
| `title`              | `WAMCast thermospheric mass-density forecast` |
| `launch_utc`         | `2025-11-11T00:00:00+00:00`                  |
| `driver_protocol`    | `observed_omni` / `swpc_forecast` / `frozen` |
| `model_variant`      | `standard_t16_ensemble_k20`                  |
| `checkpoint_paths`   | Semi-colon-separated paths                   |
| `wamcast_version`    | `0.1.0`                                      |
| `conformal_alpha`    | `0.05` (only if intervals present)           |

## Reading with xarray

```python
import xarray as xr
ds = xr.open_dataset("forecast.nc")
print(ds.attrs)
print(ds["rho_mu"].sel(horizon=48))
```
