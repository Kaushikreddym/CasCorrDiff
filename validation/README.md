# Climate Model Validation Scripts

This directory contains scripts for validating climate models against GHCNd station observations.

## Structure

- `viz_utils.py` - Shared visualization functions and metric calculations
- `validate_pr.py` - Precipitation validation script
- `validate_tasmax.py` - Maximum temperature validation script
- `outputs/` - Generated plots and figures
  - `pr/` - Precipitation validation outputs
  - `tasmax/` - Temperature validation outputs

## Usage

### Precipitation Validation
```bash
bash validation/run_pr_validation.sh
```

### Temperature Validation
```bash
bash validation/run_tasmax_validation.sh
```

**Note:** These scripts automatically activate the `sdba` conda environment before running the validation.

## Outputs

Each validation script generates the following plots:

1. **Seasonal Maps** (`seasonal_maps_*.png`)
   - 4x6 panel showing seasonal means (DJF, MAM, JJA, SON)
   - Columns: GHCNd, ERA5, Prediction (10km), ISIMIP, Prediction (1km), DWD

2. **Metric Maps** (`metric_*_*.png`)
   - RMSE
   - Spearman correlation
   - Wasserstein distance
   - Extreme bias (90th percentile)
   - *Precipitation only:* Frequency bias, POD, CSI

3. **Quantile Difference Plot** (`quantile_difference_*.png`)
   - Model vs observation differences across quantiles
   - Shows over/underestimation patterns

4. **2D Density Scatter** (`density_scatter_*.png`)
   - Station-mean scatter plots with KDE
   - Performance metrics: r, R², NSE, percent error

## Models Compared

- **ERA5 (100 km)** - Coarse input
- **CorrDiff Prediction (10 km)** - First downscaling stage
- **CorrDiff Prediction (1 km)** - Final downscaling stage
- **ISIMIP BCSD (1 km)** - Alternative bias correction method
- **MSWX (10 km)** - Target for 10km stage
- **DWD (1 km)** - Target for 1km stage

## Requirements

- xarray
- numpy
- matplotlib
- cartopy
- seaborn
- scipy
- colormaps
- datasets (local module)
