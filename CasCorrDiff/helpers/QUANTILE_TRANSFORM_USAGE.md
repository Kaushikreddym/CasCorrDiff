# Quantile Transform for Precipitation Normalization

This document explains how to use the new quantile transform feature for precipitation normalization in the MSWX-DWD dataset.

## Overview

The quantile transform is an alternative to log-transform + z-score normalization for precipitation data. It maps precipitation values to a uniform distribution using empirical quantiles learned from the data, which can be more robust for skewed distributions.

### Benefits:
- Maps precipitation to uniform [0, 1] distribution (more intuitive for neural networks)
- Robust to outliers compared to log-transform
- Better handling of zero-precipitation and extreme values
- sklearn's QuantileTransformer with `output_distribution='uniform'`

## Step 1: Calculate Statistics and Quantile Transforms

### Basic Usage (without quantile transform):
```bash
python -u helpers/calc_stats_mswxdwd.py \
    --domain germany \
    --mswx-path /beegfs/muduchuru/data/mswx \
    --hyras-path /beegfs/muduchuru/data/HYRAS_DAILY \
    --output-dir /beegfs/muduchuru/data \
    --skip-hyras \
    --n-workers 80
```

### With Quantile Transforms (recommended for precipitation):
```bash
python -u helpers/calc_stats_mswxdwd.py \
    --domain germany \
    --mswx-path /beegfs/muduchuru/data/mswx \
    --hyras-path /beegfs/muduchuru/data/HYRAS_DAILY \
    --output-dir /beegfs/muduchuru/data \
    --compute-quantile-transforms \
    --n-quantiles 1000 \
    --n-workers 80
```

### Output Files:
- `mswx_stats_germany_log.json` - Mean/std statistics for MSWX
- `hyras_stats_germany_log.json` - Mean/std statistics for HYRAS
- `mswx_stats_germany_quantile_transform.pkl` - Quantile transformer for MSWX precipitation
- `hyras_stats_germany_quantile_transform.pkl` - Quantile transformer for HYRAS precipitation

## Step 2: Use Quantile Transform in Dataset

### Without Quantile Transform (default):
```python
from datasets.mswxdwd import mswxdwd

dataset = mswxdwd(
    data_path="/path/to/data",
    input_channels=["pr", "tas", "hurs"],
    output_channels=["pr", "tas"],
    stats_mswx="/beegfs/muduchuru/data/mswx_stats_germany_log.json",
    stats_dwd="/beegfs/muduchuru/data/hyras_stats_germany_log.json",
    normalize=True,
    use_quantile_transform=False  # Use log-transform + z-score
)
```

### With Quantile Transform (new):
```python
from datasets.mswxdwd import mswxdwd

dataset = mswxdwd(
    data_path="/path/to/data",
    input_channels=["pr", "tas", "hurs"],
    output_channels=["pr", "tas"],
    stats_mswx="/beegfs/muduchuru/data/mswx_stats_germany_log.json",
    stats_dwd="/beegfs/muduchuru/data/hyras_stats_germany_log.json",
    quantile_transform_mswx="/beegfs/muduchuru/data/mswx_stats_germany_quantile_transform.pkl",
    quantile_transform_dwd="/beegfs/muduchuru/data/hyras_stats_germany_quantile_transform.pkl",
    normalize=True,
    use_quantile_transform=True  # Use quantile transform for precipitation
)
```

## Step 3: Normalization Behavior

When `use_quantile_transform=True`:

### Normalization (during training):
1. **Precipitation (pr)**: Applied quantile transform → scaled to [-1, 1]
2. **Other channels** (tas, hurs, etc.): Standard z-score normalization

### Denormalization (inference):
1. **Precipitation (pr)**: Scaled from [-1, 1] to [0, 1] → inverse quantile transform
2. **Other channels**: Reverse z-score normalization

## Command Line Options

### For `calc_stats_mswxdwd.py`:

```
--compute-quantile-transforms    Compute quantile transforms for precipitation (saves as .pkl)
--n-quantiles N                  Number of quantile points (default: 1000)
                                 Lower values = faster, less memory
                                 Higher values = smoother transformation
```

## Technical Details

- **sklearn QuantileTransformer** with:
  - `output_distribution='uniform'` - Maps to [0, 1]
  - `subsample=1e8` - Uses up to 100M samples for fitting
  - `random_state=42` - Reproducible results

- **Pickle format**: Serialized sklearn QuantileTransformer objects

## Example Usage in Training Script

```python
import argparse
from datasets.mswxdwd import mswxdwd

parser = argparse.ArgumentParser()
parser.add_argument("--use-quantile-transform", action="store_true",
                    help="Use quantile transform for precipitation")
args = parser.parse_args()

dataset = mswxdwd(
    data_path="/path/to/data",
    input_channels=["pr", "tas", "hurs"],
    output_channels=["pr", "tas"],
    stats_mswx="/beegfs/muduchuru/data/mswx_stats_germany_log.json",
    stats_dwd="/beegfs/muduchuru/data/hyras_stats_germany_log.json",
    quantile_transform_mswx=(
        "/beegfs/muduchuru/data/mswx_stats_germany_quantile_transform.pkl" 
        if args.use_quantile_transform else None
    ),
    quantile_transform_dwd=(
        "/beegfs/muduchuru/data/hyras_stats_germany_quantile_transform.pkl"
        if args.use_quantile_transform else None
    ),
    normalize=True,
    use_quantile_transform=args.use_quantile_transform
)
```

## Troubleshooting

### Issue: "No valid data found for precipitation"
- Check that precipitation files exist and contain non-zero, non-NaN values
- Verify file paths are correct

### Issue: Pickle file too large
- Reduce `--n-quantiles` (e.g., 100-500 instead of 1000)
- Larger n_quantiles = more accurate but larger file

### Issue: Quantile transform not applied
- Verify `use_quantile_transform=True` is set
- Check that pickle files are properly loaded (no errors in initialization)
- Ensure precipitation channel names match (should be "pr")

## Performance Considerations

- **Memory**: Quantile transform uses sklearn's subsample feature to handle large datasets
- **Speed**: Quantile computation is slightly slower than log-transform during statistics calculation
- **Inference**: Same speed for both approaches during normalization/denormalization

## File Structure

```
/beegfs/muduchuru/data/
├── mswx_stats_germany_log.json                    # Mean/std stats
├── mswx_stats_germany_quantile_transform.pkl      # Quantile transformer
├── hyras_stats_germany_log.json                   # Mean/std stats
└── hyras_stats_germany_quantile_transform.pkl     # Quantile transformer
```
