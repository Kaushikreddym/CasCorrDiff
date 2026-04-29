"""
Maximum temperature (tasmax) validation script.

Validates multiple temperature models against GHCNd station observations:
- ERA5 (100 km input)
- CorrDiff predictions (10 km and 1 km)
- ISIMIP BCSD (1 km)
- MSWX target (10 km)
- DWD target (1 km)

Creates:
- Seasonal maps
- Metric maps (RMSE, Spearman, Kendall, etc.)
- Quantile difference plots
- 2D density scatter plots with KDE
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import xarray as xr
import numpy as np
import colormaps as cmaps
from datasets import assign_season
from validation.viz_utils import (
    calculate_metrics,
    plot_seasonal_maps,
    plot_metric_maps,
    plot_quantile_difference,
    plot_2d_density_scatter
)

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "station_extracts_tasmax/all_stations_tasmax_2020-2023.nc"
OUTPUT_DIR = BASE_DIR / "validation/outputs/tasmax"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("MAXIMUM TEMPERATURE VALIDATION")
print("="*80)

# Load data
print("\nLoading tasmax station data...")
ds = xr.open_dataset(DATA_PATH)

print(f"Dimensions: {ds.dims}")
print(f"Variables: {list(ds.data_vars)}")
print(f"Time range: {ds.time.min().values} to {ds.time.max().values}")

# Filter stations with DWD data
print("\nFiltering stations with valid DWD (1km) data...")
dwd_has_data = ds['dwd_target_1km'].notnull().any(dim='time')
valid_stations = dwd_has_data.sum().values
print(f"Stations with DWD data: {valid_stations}/{len(ds.station)}")

ds = ds.where(dwd_has_data, drop=True)

# Extract variables
obs = ds['observations']
era5_input = ds['era5_input_100km']
pred_10km = ds['prediction_10km']
mswx_target_10km = ds['mswx_target_10km']
pred_1km = ds['prediction_1km']
dwd_target_1km = ds['dwd_target_1km']
pred_isimip = ds['prediction_isimip_bcsd']

ds_obs = ds[['lat', 'lon']].assign({'station': ds.station})

# Handle ensemble dimensions
if 'ensemble' in pred_10km.dims:
    pred_10km_mean = pred_10km.mean('ensemble')
else:
    pred_10km_mean = pred_10km

if 'ensemble' in pred_1km.dims:
    pred_1km_mean = pred_1km.mean('ensemble')
else:
    pred_1km_mean = pred_1km

pred_isimip_mean = pred_isimip

print(f"\nDataset statistics:")
print(f"  Observations mean: {obs.mean().values:.2f} °C")
print(f"  ERA5 input mean: {era5_input.mean().values:.2f} °C")
print(f"  10km prediction mean: {pred_10km_mean.mean().values:.2f} °C")
print(f"  ISIMIP prediction mean: {pred_isimip_mean.mean().values:.2f} °C")
print(f"  1km prediction mean: {pred_1km_mean.mean().values:.2f} °C")

# ============================================================================
# SEASONAL MAPS
# ============================================================================
print("\n" + "="*80)
print("Creating seasonal maps...")
print("="*80)

# Create season labels
obs_seasons = xr.DataArray(assign_season(obs.time), coords={'time': obs.time}, dims='time')
era5_input_seasons = xr.DataArray(assign_season(era5_input.time), 
                                  coords={'time': era5_input.time}, dims='time')
pred_10km_seasons = xr.DataArray(assign_season(pred_10km_mean.time), 
                                coords={'time': pred_10km_mean.time}, dims='time')
pred_1km_seasons = xr.DataArray(assign_season(pred_1km_mean.time), 
                               coords={'time': pred_1km_mean.time}, dims='time')
mswx_10km_seasons = xr.DataArray(assign_season(mswx_target_10km.time), 
                                coords={'time': mswx_target_10km.time}, dims='time')
dwd_1km_seasons = xr.DataArray(assign_season(dwd_target_1km.time), 
                              coords={'time': dwd_target_1km.time}, dims='time')
pred_isimip_seasons = xr.DataArray(assign_season(pred_isimip_mean.time), 
                                  coords={'time': pred_isimip_mean.time}, dims='time')

# Calculate seasonal means
obs_seasonal_raw = obs.groupby(obs_seasons).mean('time')
era5_input_seasonal_raw = era5_input.groupby(era5_input_seasons).mean('time')
pred_10km_seasonal_raw = pred_10km_mean.groupby(pred_10km_seasons).mean('time')
pred_1km_seasonal_raw = pred_1km_mean.groupby(pred_1km_seasons).mean('time')
mswx_10km_seasonal_raw = mswx_target_10km.groupby(mswx_10km_seasons).mean('time')
dwd_1km_seasonal_raw = dwd_target_1km.groupby(dwd_1km_seasons).mean('time')
pred_isimip_seasonal_raw = pred_isimip_mean.groupby(pred_isimip_seasons).mean('time')

# Apply 80% validity mask
print("Applying 80% valid data mask...")
valid_mask = (obs.notnull().groupby(obs_seasons).sum('time') / 
              obs_seasons.groupby(obs_seasons).count()) >= 0.8

obs_seasonal = obs_seasonal_raw.where(valid_mask).compute()
era5_input_seasonal = era5_input_seasonal_raw.where(valid_mask).compute()
pred_10km_seasonal = pred_10km_seasonal_raw.where(valid_mask).compute()
pred_1km_seasonal = pred_1km_seasonal_raw.where(valid_mask).compute()
mswx_10km_seasonal = mswx_10km_seasonal_raw.where(valid_mask).compute()
dwd_1km_seasonal = dwd_1km_seasonal_raw.where(valid_mask).compute()
pred_isimip_seasonal = pred_isimip_seasonal_raw.where(valid_mask).compute()

# Plot seasonal maps
datasets = [obs_seasonal, era5_input_seasonal, pred_10km_seasonal, 
            pred_isimip_seasonal, pred_1km_seasonal, dwd_1km_seasonal]
col_titles = ['GHCNd', 'ERA5 (100 km)', 'Prediction (10 km)', 
              'ISIMIP3BASD (10 km)', 'Prediction (1 km)', 'DWD (Target-1km)']

plot_seasonal_maps(datasets, col_titles, ds_obs, 
                  output_path=OUTPUT_DIR / "seasonal_maps_tasmax.png",
                  variable_name='tasmax', variable_units='°C')
print(f"Saved: {OUTPUT_DIR / 'seasonal_maps_tasmax.png'}")

# ============================================================================
# CALCULATE METRICS
# ============================================================================
print("\n" + "="*80)
print("Calculating metrics...")
print("="*80)

# Align datasets
common_times = obs.time.values
era5_input_aligned = era5_input.sel(time=common_times, method='nearest').transpose('station', 'time')
pred_10km_aligned = pred_10km_mean.sel(time=common_times, method='nearest').transpose('station', 'time')
pred_isimip_aligned = pred_isimip_mean.sel(time=common_times, method='nearest').transpose('station', 'time')
pred_1km_aligned = pred_1km_mean.sel(time=common_times, method='nearest').transpose('station', 'time')

# Calculate metrics (no wet threshold for temperature)
print("Calculating metrics for each model...")
metrics_era5 = calculate_metrics(obs.values, era5_input_aligned.values, wet_threshold=-999)
metrics_10km = calculate_metrics(obs.values, pred_10km_aligned.values, wet_threshold=-999)
metrics_isimip = calculate_metrics(obs.values, pred_isimip_aligned.values, wet_threshold=-999)
metrics_1km = calculate_metrics(obs.values, pred_1km_aligned.values, wet_threshold=-999)

model_names = ['ERA5 (100 km)', 'Prediction (10 km)', 'ISIMIP3BASD (10 km)', 'Prediction (1 km)']
all_metrics = [metrics_era5, metrics_10km, metrics_isimip, metrics_1km]

print("\nValid stations per model:")
for name, metrics in zip(model_names, all_metrics):
    print(f"  {name}: {np.sum(~np.isnan(metrics['rmse']))} stations")

# ============================================================================
# METRIC MAPS
# ============================================================================
print("\n" + "="*80)
print("Creating metric maps...")
print("="*80)

# Temperature-specific metrics (no wet-day metrics)
all_metric_info = [
    {'name': 'RMSE (°C)', 'key': 'rmse', 'cmap': cmaps.agsunset_r},
    {'name': 'Spearman ρ', 'key': 'spearman', 'cmap': cmaps.agsunset_r, 
     'vmin': 0.6, 'vmax': 0.95},
    {'name': 'Wasserstein Distance (°C)', 'key': 'wasserstein', 
     'cmap': cmaps.agsunset_r},
    {'name': 'Extreme Bias (°C)', 'key': 'extreme_bias', 
     'cmap': cmaps.precip_diff_12lev, 'center': 0.0}
]

for metric_info in all_metric_info:
    metric_key = metric_info['key']
    output_file = OUTPUT_DIR / f"metric_{metric_key}_tasmax.png"
    
    print(f"Creating {metric_info['name']} map...")
    plot_metric_maps(all_metrics, model_names, metric_info, ds_obs, 
                    output_path=output_file)
    print(f"Saved: {output_file}")

# ============================================================================
# QUANTILE DIFFERENCE PLOT
# ============================================================================
print("\n" + "="*80)
print("Creating quantile difference plot...")
print("="*80)

model_data_list = [
    ('ERA5 (100 km)', era5_input_aligned, 'o', '#8c564b'),
    ('Prediction (10 km)', pred_10km_aligned, 'o', '#1f77b4'),
    ('Prediction (1 km)', pred_1km_aligned, 's', '#5a9bd4'),
    ('ISIMIP3BASD (10 km)', pred_isimip_aligned, '^', '#2ca02c'),
]

plot_quantile_difference(obs, model_data_list, 
                        output_path=OUTPUT_DIR / "quantile_difference_tasmax.png",
                        variable_name='tasmax', variable_units='°C')
print(f"Saved: {OUTPUT_DIR / 'quantile_difference_tasmax.png'}")

# ============================================================================
# 2D DENSITY SCATTER PLOTS
# ============================================================================
print("\n" + "="*80)
print("Creating 2D density scatter plots...")
print("="*80)

# Include MSWX and DWD for scatter plots
mswx_target_aligned = mswx_target_10km.sel(time=common_times, method='nearest').transpose('station', 'time')
dwd_target_aligned = dwd_target_1km.sel(time=common_times, method='nearest').transpose('station', 'time')

model_data_list_scatter = [
    ('ERA5 (100 km)', era5_input_aligned, '#8c564b'),
    ('Prediction (10 km)', pred_10km_aligned, '#1f77b4'),
    ('Prediction (1 km)', pred_1km_aligned, '#5a9bd4'),
    ('ISIMIP3BASD (10 km)', pred_isimip_aligned, '#2ca02c'),
    ('MSWX (10 km)', mswx_target_aligned, '#7f7f7f'),
    ('DWD (1 km)', dwd_target_aligned, '#505050'),
]

plot_2d_density_scatter(obs, model_data_list_scatter,
                       output_path=OUTPUT_DIR / "density_scatter_tasmax.png",
                       variable_name='tasmax', variable_units='°C')
print(f"Saved: {OUTPUT_DIR / 'density_scatter_tasmax.png'}")

print("\n" + "="*80)
print("MAXIMUM TEMPERATURE VALIDATION COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
