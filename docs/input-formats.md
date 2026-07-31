# Input formats

## ML-ready HDF5

The `--h5` file is a preprocessed WAM-IPE archive with 10-minute cadence. Required datasets:

| Dataset  | Shape                        | dtype     | Notes |
|----------|------------------------------|-----------|-------|
| `rho`    | `(N, 51, 91, 90)`            | float32   | Mass density on WAM-IPE pressure levels 10-50, lat 91 rows (poles + 89 interior), lon 90 cols |
| `lat`    | `(91,)`                      | float32   | Degrees north, -90..90 |
| `lon`    | `(90,)`                      | float32   | Degrees east, 0..358 |
| `time`   | `(N,)`                       | int64     | UTC epoch nanoseconds |
| `doy`    | `(N,)`                       | float32   | Day of year with fractional hour |
| `driver` | `(N, 2)`                     | float32   | **Column 0 = Kp, column 1 = F10.7** |

The model only uses pressure levels 10-50 (41 levels total = "upper thermosphere"), latitudes 1..89 (excluding polar rings), all 90 longitudes.

## OMNI2 CSV (observed drivers, `--driver-protocol observed_omni`)

Plain CSV with three columns:

```
time_utc,Kp,F107
2025-11-10T00:00:00Z,2.7,164.2
2025-11-11T03:00:00Z,8.7,168.5
...
```

- `time_utc` may be tz-aware (`Z` suffix) or naive UTC. Naive is localized to UTC.
- Values sparser than 10 minutes (3-hour Kp, daily F10.7) are forward-filled.

## SWPC RSGA JSON (operational forecast, `--driver-protocol swpc_forecast`)

A simplified daily-step bulletin:

```json
{
  "issued_utc": "2025-11-10T21:30:00Z",
  "valid_from_utc": "2025-11-11T00:00:00Z",
  "kp_forecast_daily": [
    {"date": "2025-11-11", "kp_max": 7},
    {"date": "2025-11-12", "kp_max": 5}
  ],
  "f107_forecast_daily": [
    {"date": "2025-11-11", "f107": 168},
    {"date": "2025-11-12", "f107": 170}
  ]
}
```

Each daily value is held constant across its UT day. **The caller is responsible for the leak-safe 22:00 UTC issue-time rule** used by the paper's operational protocol — do not include forecast days that were not yet issued at the launch time. The research repo's `scripts/build_swpc_forecast_sidecar.py` implements the full rule if reproducibility of the paper's operational headline is required.

## Frozen scalars (ablation, `--driver-protocol frozen`)

No file needed; pass `--frozen-kp <value> --frozen-f107 <value>` for constant drivers across the rollout window.
