# CasCorrDiff: Cascading Correction Diffusion Models for Atmospheric Downscaling

A comprehensive framework for generative downscaling and climate data correction using diffusion models, combining deterministic regression and stochastic diffusion approaches for km-scale weather prediction and extreme event analysis.

## Overview

CasCorrDiff applies cascading correction techniques with diffusion models to downscale coarse-resolution atmospheric data (e.g., ERA5) to high-resolution predictions. The framework includes:

- **Regression-Diffusion Two-Step Training**: Initial deterministic regression followed by diffusion-based refinement
- **Multi-Dataset Support**: HRRR, GEFS, ERA5, MSWX, Taiwan CWB, and custom datasets
- **Extreme Events Analysis**: Specialized analysis for compound extremes and spatial patterns
- **Validation Tools**: Comprehensive validation against in-situ observations
- **Inference Pipeline**: Efficient generation and post-processing utilities

## Repository Structure

```
CasCorrDiff/
├── CasCorrDiff/              # Main package
│   ├── datasets/             # Data loading utilities
│   ├── inference/            # Prediction and sampling
│   ├── helpers/              # Helper functions
│   ├── obs/                  # Observational data handling
│   ├── conf/                 # Configuration files (YAML)
│   └── train.py              # Training script
├── datasets/                 # Custom dataset implementations
├── diffusion_output_processing/  # Post-processing and analysis
├── extremes/                 # Extreme events analysis
├── validation/               # Validation scripts and utilities
├── station_extracts/         # Station-level data extraction
└── extract_stations_parallel.py  # Parallel station extraction
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd CasCorrDiff

# Install dependencies
pip install -r CasCorrDiff/requirements.txt
```

### Training

Configure training through YAML files in `CasCorrDiff/conf/`:

```bash
# Train regression model
cd CasCorrDiff
python train.py --config-name=config_training_custom.yaml

# Or override parameters from command line
python train.py --config-name=config_training_custom.yaml ++training.hp.total_batch_size=64
```

### Inference

Generate predictions using trained models:

```bash
python generate.py --config-name=config_generate_custom.yaml
```

### Validation

Validate model predictions against observations:

```bash
cd validation/
python validate_pr.py      # Precipitation validation
python validate_tasmax.py  # Temperature validation
```

## Key Features

- **Modular Design**: Separate components for datasets, models, inference, and post-processing
- **Configuration-Driven**: YAML-based configuration system using Hydra for reproducibility
- **Multi-Variable Support**: Temperature (TASMAX/TASMIN) and precipitation
- **Spatial Analysis**: Pattern analysis, clustering, and extreme event detection
- **Performance Monitoring**: Integration with Weights & Biases (W&B) for experiment tracking

## Data Requirements

The framework supports multiple datasets:

- **HRRR**: High-Resolution Rapid Refresh (US domain)
- **GEFS**: Global Ensemble Forecast System
- **ERA5**: ECMWF Reanalysis v5
- **MSWX**: Multisource Weighted-Ensemble Precipitation
- **Taiwan CWB**: Central Weather Bureau data
- **Custom**: User-defined datasets via custom data loaders

## Documentation

- [CasCorrDiff Main README](CasCorrDiff/README.md) - Detailed model documentation
- [Diffusion Output Processing README](diffusion_output_processing/README.md) - Analysis and post-processing
- [Extremes Analysis README](extremes/ANALYSIS_STRUCTURE.md) - Extreme events methodology
- [Validation README](validation/README.md) - Validation procedures

## Configuration

Training and generation configurations are located in `CasCorrDiff/conf/`:

- **Training**: `config_training_*.yaml` files for different datasets
- **Generation**: `config_generate_*.yaml` files for prediction
- **Customization**: Modify YAML files or use command-line overrides

## Requirements

- Python 3.9+
- CUDA-capable GPU (for training)
- Dependencies listed in `CasCorrDiff/requirements.txt`:
  - PyTorch/Lightning
  - Hydra
  - xarray/netCDF4
  - Weights & Biases
  - scikit-learn, scipy, numba

See [CasCorrDiff/requirements.txt](CasCorrDiff/requirements.txt) for full dependencies.

## Usage Examples

### Extract Station Data
```bash
python extract_stations_parallel.py --dataset era5 --output station_extracts/
```

### Compare with Observations
See notebooks in `diffusion_output_processing/`:
- `compare_bcsd_diffusion_to_obs.py`
- `compare_to_obs_insitu.ipynb`

### Spatial Pattern Analysis
```bash
# See notebooks in diffusion_output_processing/
jupyter notebook diffusion_output_processing/diffusion_spatial_patterns.ipynb
```

## Citation

If you use CasCorrDiff in your research, please cite:

```bibtex
@software{muduchuru2026cascorrdiff,
  author = {Muduchuru, Kaushik},
  title = {CasCorrDiff: Cascading Correction Diffusion Models for Atmospheric Downscaling},
  year = {2026},
  url = {https://github.com/[username]/CasCorrDiff},
  note = {Software available at https://zenodo.org/records/[record-id]}
}
```

## License

This project is licensed under the [LICENSE](LICENSE) - see file for details.

## Acknowledgments

- Built with physicsnemo, PyTorch, PyTorch Lightning, and Hydra
- Data sources: ERA5, MSWX, DWD, GHCNd
