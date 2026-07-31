# Quickstart

## 1. Install

```bash
pip install wamcast
```

## 2. Download demo artifacts

```bash
wamcast download-artifacts --name demo_launch_h5 --name train_stats --name ensemble_t16_member_00
```

Files land in `~/.cache/wamcast/`.

## 3. Run a forecast

```bash
wamcast forecast \
    --ckpt ~/.cache/wamcast/ensemble_t16_member_00.ckpt \
    --h5 ~/.cache/wamcast/demo_launch.h5 \
    --stats ~/.cache/wamcast/train_stats.npz \
    --launch 2025-11-11T00:00:00Z \
    --driver-protocol frozen --frozen-kp 8.7 --frozen-f107 170 \
    --out forecast.nc
```

## 4. Read the forecast

```python
import xarray as xr

ds = xr.open_dataset("forecast.nc")
print(ds)
# Dimensions: (horizon: 5, level: 41, latitude: 89, longitude: 90)
# Data variables:
#     rho_mu     (horizon, level, latitude, longitude) float32
#     rho_sigma  (horizon, level, latitude, longitude) float32

# Mean forecast at +48h, altitude bin ~470 km, over the equatorial belt
mu_48h = ds["rho_mu"].sel(horizon=48).mean(dim="level").sel(
    latitude=slice(-10, 10)).mean(dim=("latitude", "longitude"))
print(f"+48h mean equatorial mu = {float(mu_48h):.3f}")
```

## 5. With calibrated uncertainty

```bash
# Fit a calibrator on cal-set predictions (one-time)
wamcast calibrate \
    --cal-preds cal_storm_1.npz cal_storm_2.npz \
    --alpha 0.05 --out calibrator.npz

# Rerun forecast with intervals
wamcast forecast \
    --ckpt ~/.cache/wamcast/ensemble_t16_member_*.ckpt \
    --h5 ~/.cache/wamcast/demo_launch.h5 \
    --stats ~/.cache/wamcast/train_stats.npz \
    --launch 2025-11-11T00:00:00Z \
    --driver-protocol frozen --frozen-kp 8.7 --frozen-f107 170 \
    --calibrator calibrator.npz \
    --out forecast_with_intervals.nc
```

The output now includes `rho_lo` and `rho_hi` fields with 95% coverage (α=0.05).

## Library API

```python
from wamcast.dataset import ForecastInputs
from wamcast.drivers import frozen_drivers
from wamcast.model import load_wamcast_from_checkpoint
from wamcast.rollout import rollout

drivers = frozen_drivers(kp=8.7, f107=170.0,
                         start="2025-11-11T00:00:00Z",
                         end="2025-11-13T00:00:00Z")
inputs = ForecastInputs.from_launch(
    h5_path="demo_launch.h5", stats_path="train_stats.npz",
    launch_utc="2025-11-11T00:00:00Z", drivers=drivers,
)
model = load_wamcast_from_checkpoint("ensemble_t16_member_00.ckpt").cuda().eval()
forecast = rollout(model, inputs, drivers=drivers,
                   h5_path="demo_launch.h5", stats_path="train_stats.npz")

# forecast.mu shape: (5, 1, 41, 89, 90) — (horizon, batch, level, lat, lon)
```
