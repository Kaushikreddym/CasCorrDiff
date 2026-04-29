# CasCorrDiff Datasets Package

A Python package for loading and processing weather model data and extracting it at GHCN observation station locations.

## Overview

This package provides three main classes for handling different model datasets:
- **ERA5MSWX**: 10km resolution ERA5-MSWX model data
- **MSWXDWD**: 1km resolution MSWX-DWD model data
- **ISIMIP_ERA5**: ISIMIP BCSD-corrected ERA5 data

## Installation

The package can be imported directly since it's part of the CasCorrDiff project:

```python
from datasets import ERA5MSWX, MSWXDWD, ISIMIP_ERA5
```

## Classes

### ERA5MSWX

Handles 10km resolution ERA5-MSWX model data.

**Methods:**
- `__init__(base_path)`: Initialize with path to data directory
- `load(years, chunks)`: Load data for specified years into xarray datasets
- `extract_GHCN(ds_obs, variable)`: Extract data at GHCN station locations

**Example:**
```python
from datasets import ERA5MSWX, load_GHCN, extract_dataset_metadata

# Initialize and load model data
era5 = ERA5MSWX(base_path="/path/to/data/")
era5.load(years=[2019, 2020, 2021])

# Load GHCN observations using the simple load_GHCN function
metadata = extract_dataset_metadata(era5.inv)
ds_obs = load_GHCN(metadata=metadata, max_stations=50)

# Extract precipitation at stations
results = era5.extract_GHCN(ds_obs, variable='pr')

# Access results
observations = results['observations']
model_input = results['model_input']
model_prediction = results['model_prediction']
model_truth = results['model_truth']
```

### MSWXDWD

Handles 1km resolution MSWX-DWD model data.

**Methods:**
- `__init__(base_path)`: Initialize with path to data directory
- `load(years, chunks)`: Load data for specified years into xarray datasets
- `extract_GHCN(ds_obs, variable)`: Extract data at GHCN station locations

**Example:**
```python
from datasets import MSWXDWD, load_GHCN, extract_dataset_metadata

# Initialize and load model data
mswx = MSWXDWD(base_path="/path/to/data/")
mswx.load(years=[2019, 2020])

# Load GHCN observations using the simple load_GHCN function
metadata = extract_dataset_metadata(mswx.inv)
ds_obs = load_GHCN(metadata=metadata, max_stations=30)

# Extract maximum temperature at stations
results = mswx.extract_GHCN(ds_obs, variable='tasmax')

# Access results
observations = results['observations']
model_input = results['model_input']
model_prediction = results['model_prediction']
model_truth = results['model_truth']
```

### ISIMIP_ERA5

Handles ISIMIP BCSD-corrected ERA5 data with automatic NaN dropping.

**Key Features:**
- Automatically drops NaN-only rows/columns from lat/lon dimensions
- Uses time-invariant coordinates (1D lat/lon arrays)
- Provides both BCSD predictions and observations

**Methods:**
- `__init__(base_path)`: Initialize with path to BCSD output directory
- `load(years, chunks)`: Load BCSD data for specified years
- `extract_GHCN(ds_obs, variable)`: Extract data at GHCN station locations

**Example:**
```python
from datasets import ISIMIP_ERA5, load_GHCN, extract_dataset_metadata

# Initialize and load ISIMIP BCSD data
isimip = ISIMIP_ERA5()
isimip.load(years=[2020, 2021, 2022])

# Load GHCN observations
metadata = extract_dataset_metadata(isimip.inv)
ds_obs = load_GHCN(metadata=metadata, max_stations=100)

# Extract precipitation at stations
results = isimip.extract_GHCN(ds_obs, variable='pr')

# Access results (note: no 'model_input' for BCSD)
observations = results['observations']
model_prediction = results['model_prediction']  # BCSD prediction
model_truth = results['model_truth']  # BCSD observations
```

## Supported Variables

- `pr`: Precipitation
- `tas`: Mean temperature
- `tasmin`: Minimum temperature
- `tasmax`: Maximum temperature

## Method Details

### load(years, chunks)

Loads model data for specified years into xarray datasets.

**Parameters:**
- `years` (list of int): Years to load data for
- `chunks` (dict, optional): Chunking specification for dask arrays. Default: `{"time": 10}`

**Returns:**
- Self (for method chaining)

**Attributes set:**
- `self.input`: Model input data
- `self.prediction`: Model prediction data
- `self.truth`: Truth/target data
- `self.inv`: Invariant/coordinate data
- `self.years`: List of loaded years

### extract_GHCN(ds_obs, variable)

Extracts variable data at GHCN station locations.

**Parameters:**
- `ds_obs` (xarray.Dataset): GHCN observations dataset (load using `load_GHCN()` function)
- `variable` (str): Variable name to extract

**Returns:**
- dict with keys:
  - `'observations'`: xarray.Dataset with GHCN observations
  - `'model_input'`: xarray.Dataset with model input at stations
  - `'model_prediction'`: xarray.Dataset with model predictions at stations
  - `'model_truth'`: xarray.Dataset with truth data at stations
  - `'indices'`: tuple of (y_indices, x_indices) for station locations
  - `'variable'`: name of extracted variable

## Helper Functions

### load_GHCN(metadata=None, shapefile=None, ...)

Simple function to load GHCN observations.

**Parameters:**
- `metadata` (dict, optional): Domain metadata from `extract_dataset_metadata()`
- `shapefile` (str, optional): Path to shapefile for spatial filtering (coming soon)
- `ghcn_base_path` (str): Path to GHCN data directory. Default: `"/data01/FDS/muduchuru/Atmos/GHCN"`
- `elements` (list): GHCN elements to load. Default: `['TMAX', 'TMIN', 'TAVG', 'PRCP']`
- `max_stations` (int): Maximum number of stations to process. Default: 100
- `verbose` (bool): Print progress information. Default: True

**Returns:**
- xarray.Dataset with GHCN observations

**Note:** Either `metadata` or `shapefile` must be provided. Shapefile support is planned for future release.

**Example:**
```python
from datasets import load_GHCN, extract_dataset_metadata

# Get metadata from model
metadata = extract_dataset_metadata(model.inv)

# Load GHCN observations
ds_obs = load_GHCN(
    metadata=metadata,
    elements=['PRCP', 'TMAX'],
    max_stations=50,
    verbose=True
)
```

## Advanced Usage

### Loading GHCN Once for Multiple Extractions

The efficient pattern is to load GHCN observations once and reuse them:

```python
from datasets import ERA5MSWX, MSWXDWD, load_GHCN, extract_dataset_metadata

# Load model data
era5 = ERA5MSWX().load(years=[2019, 2020])
mswx = MSWXDWD().load(years=[2019, 2020])

# Load GHCN observations once using the simple function
metadata = extract_dataset_metadata(era5.inv)
ds_obs = load_GHCN(
    metadata=metadata,
    elements=['PRCP', 'TMAX', 'TMIN', 'TAVG'],
    max_stations=50,
    verbose=True
)

# Extract multiple variables from both models using same observations
variables = ['pr', 'tas', 'tasmin', 'tasmax']

era5_results = {var: era5.extract_GHCN(ds_obs, var) for var in variables}
mswx_results = {var: mswx.extract_GHCN(ds_obs, var) for var in variables}
```

### Using Shapefile for Spatial Filtering (Coming Soon)

```python
# Future feature - not yet implemented
ds_obs = load_GHCN(
    shapefile="/path/to/region.shp",
    elements=['PRCP', 'TAVG'],
    max_stations=100
)
```

### Custom Paths

```python
# Custom model path
era5 = ERA5MSWX(base_path="/custom/path/")
era5.load(years=[2018])

# Custom GHCN path
metadata = extract_dataset_metadata(era5.inv)
ds_obs = load_GHCN(
    metadata=metadata,
    ghcn_base_path="/custom/ghcn/path/",
    elements=['PRCP', 'TAVG'],
    max_stations=100
)
results = era5.extract_GHCN(ds_obs, variable='tas')
```

### Custom Chunking

```python
# For larger datasets, adjust chunking
mswx = MSWXDWD()
mswx.load(years=[2015, 2016, 2017, 2018], chunks={"time": 20})
```

## Requirements

- xarray
- numpy
- pandas
- scipy
- dask
- sklearn
- matplotlib
- cartopy

## File Structure

```
datasets/
├── __init__.py           # Package initialization
├── data.py              # Main classes and functions
├── example_usage.py     # Usage examples
└── README.md           # This file
```
