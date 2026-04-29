"""
Quick test of ISIMIP integration with station extraction
"""

import warnings
warnings.filterwarnings('ignore')

import xarray as xr
import numpy as np
from datasets import (
    ISIMIP_ERA5,
    load_GHCN,
    extract_dataset_metadata,
    build_kdtree,
    query_station_indices,
    extract_at_indices,
)

print("=" * 80)
print("TESTING ISIMIP INTEGRATION")
print("=" * 80)

# Load ISIMIP data
isimip = ISIMIP_ERA5()
isimip.load([2022], chunks={"time": 365})

# Load observations
metadata = extract_dataset_metadata(isimip.inv)
print(f"\nMetadata extracted:")
print(f"  Domain: {metadata['lat_min']:.2f}°N-{metadata['lat_max']:.2f}°N, {metadata['lon_min']:.2f}°E-{metadata['lon_max']:.2f}°E")

ds_obs = load_GHCN(metadata=metadata, elements=['PRCP'], max_stations=50, verbose=False)
print(f"\nLoaded {len(ds_obs.station)} stations")

# Build KDTree
kdtree, shape = build_kdtree(isimip.inv)
print(f"\nKDTree built with shape: {shape}")

# Query stations
y_idx, x_idx = query_station_indices(kdtree, shape, ds_obs)
print(f"Station indices found: y={len(y_idx)}, x={len(x_idx)}")

# Extract data
pred = extract_at_indices(isimip.prediction['pr'], y_idx, x_idx)
obs = extract_at_indices(isimip.truth['pr'], y_idx, x_idx)

print(f"\nExtracted data:")
print(f"  Prediction shape: {pred.shape}")
print(f"  Truth shape: {obs.shape}")

# Compute stats
pred_computed = pred.compute()
obs_computed = obs.compute()

print(f"\nSample statistics:")
print(f"  BCSD prediction mean: {pred_computed.mean().values:.2f} mm/day")
print(f"  BCSD truth mean: {obs_computed.mean().values:.2f} mm/day")

print("\n" + "=" * 80)
print("✅ ISIMIP INTEGRATION TEST PASSED")
print("=" * 80)
