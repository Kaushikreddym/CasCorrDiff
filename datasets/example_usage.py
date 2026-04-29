"""
Example usage of the datasets package with ERA5MSWX and MSWXDWD classes

This demonstrates how to:
1. Load GHCN observations once using the simple load_GHCN function
2. Load model data for specific years
3. Extract model data at GHCN station locations for specific variables
"""

from datasets import (
    ERA5MSWX, 
    MSWXDWD,
    load_GHCN,
    extract_dataset_metadata
)

# ==================== Step 1: Load Model Data ====================

# First, load model data to get metadata
era5 = ERA5MSWX(base_path="/beegfs/muduchuru/codes/physicsnemo/examples/weather/corrdiff/")
era5.load(years=[2019, 2020, 2021])

# ==================== Step 2: Load GHCN Observations (Once) ====================

# Extract metadata from the model
metadata = extract_dataset_metadata(era5.inv)

# Load GHCN observations using the simple load_GHCN function
ds_obs = load_GHCN(
    metadata=metadata,  # Can also use shapefile=<path> in the future
    elements=['PRCP', 'TMAX', 'TMIN', 'TAVG'],
    max_stations=50,
    verbose=True
)

print(f"\nLoaded GHCN observations:")
print(f"  Stations: {len(ds_obs.station)}")
print(f"  Time range: {ds_obs.time.min().values} to {ds_obs.time.max().values}")
print(f"  Variables: {list(ds_obs.data_vars)}")

# ==================== Step 2: Extract from ERA5MSWX (10km) ====================

# Extract precipitation at GHCN stations
results_pr = era5.extract_GHCN(ds_obs, variable='pr')

# Access the results
obs_pr = results_pr['observations']
model_input_pr = results_pr['model_input']
model_prediction_pr = results_pr['model_prediction']

print(f"\nERA5MSWX Precipitation Results:")
print(f"  Observations shape: {obs_pr.pr.shape}")
print(f"  Model input shape: {model_input_pr.pr.shape}")
print(f"  Model prediction shape: {model_prediction_pr.pr.shape}")

# Extract temperature at the same GHCN stations
results_tas = era5.extract_GHCN(ds_obs, variable='tas')

print(f"\nERA5MSWX Temperature Results:")
print(f"  Observations shape: {results_tas['observations'].tas.shape}")
print(f"  Model prediction shape: {results_tas['model_prediction'].tas.shape}")

# ==================== Step 3: Extract from MSWXDWD (1km) ====================

# Initialize and load MSWXDWD data
mswx = MSWXDWD(base_path="/beegfs/muduchuru/codes/physicsnemo/examples/weather/corrdiff/")
mswx.load(years=[2019, 2020])

# Extract maximum temperature at the same GHCN stations
results_tasmax = mswx.extract_GHCN(ds_obs, variable='tasmax')

# Access the results (MSWXDWD includes truth data)
obs_tasmax = results_tasmax['observations']
model_input_tasmax = results_tasmax['model_input']
model_prediction_tasmax = results_tasmax['model_prediction']
model_truth_tasmax = results_tasmax['model_truth']  # Unique to MSWXDWD

print(f"\nMSWXDWD Maximum Temperature Results:")
print(f"  Observations shape: {obs_tasmax.tasmax.shape}")
print(f"  Model prediction shape: {model_prediction_tasmax.tasmax.shape}")
print(f"  Model truth shape: {model_truth_tasmax.tasmax.shape}")

# ==================== Example 4: Multiple Variables ====================

# Extract all variables for both models using the same GHCN observations
variables = ['pr', 'tas', 'tasmin', 'tasmax']

print(f"\n=== Extracting all variables from ERA5MSWX ===")
era5_results = {}
for var in variables:
    era5_results[var] = era5.extract_GHCN(ds_obs, variable=var)
    print(f"  {var}: extracted at {len(era5_results[var]['observations'].station)} stations")

print(f"\n=== Extracting all variables from MSWXDWD ===")
mswx_results = {}
for var in variables:
    mswx_results[var] = mswx.extract_GHCN(ds_obs, variable=var)
    print(f"  {var}: extracted at {len(mswx_results[var]['observations'].station)} stations")
