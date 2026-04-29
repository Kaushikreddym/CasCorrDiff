# Extremes Analysis Structure

## Overview
The extremes analysis notebooks (`spatial_extremes_analysis.ipynb` and `compound_extremes_analysis.ipynb`) analyze precipitation and temperature extremes across multiple datasets with consistent model-reference comparisons.

---

## Datasets Used

### Model and Reference Pairs (across both notebooks):

#### 1. **10 km Resolution (ERA5-MSWX)**
- **Model 1 - ERA5 Input (~100 km)** vs **Reference: MSWX (10 km)**
  - ERA5 coarse-resolution reanalysis input
  - Compared against high-resolution MSWX observations
  
- **Model 2 - CasCorrDiff Prediction (10 km)** vs **Reference: MSWX (10 km)**
  - 10km diffusion model prediction
  - Compared against MSWX truth

#### 2. **ISIMIP3BASD Global Climate Model**
- **Model 3 - ISIMIP3BASD (BCSD, ~10 km)** vs **Reference: MSWX (10 km)**
  - Regridded from lat/lon to MSWX grid using xesmf (bilinear interpolation)
  - Compared against MSWX truth for consistency

---

## spatial_extremes_analysis.ipynb

### Purpose
Analyzes spatial patterns of precipitation extremes across the domain.

### Key Analyses:

#### 1. **Wet Days Frequency**
- Threshold: 10 mm/day
- Metric: Frequency of wet days (%)
- Outputs:
  - Spatial maps showing wet day frequency for each dataset
  - Bias maps (model - reference) for each comparison
  - Visualization: 2-row plot with Lambert Conformal projection
    - Row 1: Wet day frequency for all datasets
    - Row 2: Bias against reference (ERA5 vs MSWX, Prediction vs MSWX, ISIMIP vs MSWX)

#### 2. **Additional Extreme Indices** (defined, ready for use)
- **CDD** (Consecutive Dry Days): Max consecutive days with pr < 1 mm
- **CWD** (Consecutive Wet Days): Max consecutive days with pr ≥ 1 mm
- **R10mm**: Number of heavy precipitation days (≥ 10 mm)
- **R20mm**: Number of very heavy precipitation days (≥ 20 mm)
- **Rx1day**: Maximum 1-day precipitation
- **Rx5day**: Maximum consecutive 5-day precipitation

### Visualizations:
- **wet_days_frequency_bias.png**: Combined spatial comparison with:
  - Common colorbars using precip2_17lev (frequency) and precip_diff_12lev (bias)
  - Lambert Conformal projection for Germany
  - Coastlines and borders

---

## compound_extremes_analysis.ipynb

### Purpose
Analyzes relationships between temperature and precipitation extremes and their joint distributions.

### Key Features:

#### 1. **Compound Extreme Definitions** (functions defined)
- **Hot-Dry Events**: tasmax > P90 AND pr < P5
- **Hot-Wet Events**: tasmax > P90 AND pr > P95
- **Cold-Wet Events**: tasmax < P10 AND pr > P95

#### 2. **Joint Distribution Analysis** ✅ NEW
Analyzes the joint distribution of temperature and precipitation for each model-reference pair:

**Computed for:**
- ERA5 Input vs MSWX Truth
- CasCorrDiff Prediction vs MSWX Truth
- ISIMIP3BASD vs MSWX Truth

**Metrics:**
- 2D histograms showing joint frequency distribution
- Temperature range (K) and precipitation range (mm/day)
- Distribution peaks (modes)
- Difference analysis (Model - Reference counts)

**Visualization: joint_distribution_differences.png**
- 3 columns per model-reference pair:
  1. **Reference**: Joint distribution histogram (YlOrRd colormap)
  2. **Model**: Joint distribution histogram (YlOrRd colormap)
  3. **Difference**: Bias in joint distribution (precip_diff_12lev colormap)
- Shows where model over/underestimates co-occurrence of extreme temperature and precipitation

#### 3. **Statistical Summaries**
- Temperature and precipitation ranges for model and reference
- Peak location of joint distribution
- Total frequency counts
- Maximum and mean differences

**Output: joint_distribution_summary.csv**

---

## Data Processing Flow

### 1. **Loading**
```
ERA5MSWX (10km)       ← Input: ERA5
                      ← Prediction: CasCorrDiff (10km)
                      ← Truth: MSWX observations

MSWXDWD (1km)         ← Prediction: CasCorrDiff (1km)
                      ← Truth: DWD observations

ISIMIP3BASD           ← Prediction: BCSD downscaled GCM
                      ← Regridded to MSWX grid
                      ← Truth: MSWX observations
```

### 2. **Regridding**
- ISIMIP data (lat/lon grid) → MSWX grid (y/x grid)
- Method: xesmf bilinear interpolation
- Purpose: Consistent spatial comparison across datasets

### 3. **Threshold Calculation**
- Percentile-based thresholds computed per dataset per variable
- Temperature high (P90), low (P10)
- Precipitation high (P95), low (P5)

### 4. **Analysis**
- Spatial indices computed at each grid point
- Spatial means used for joint distribution (more stable)
- Bias computed as (model - reference)

---

## Configuration Parameters

### Spatial Domain (Germany)
```python
LON: 5.5° E to 15.5° E
LAT: 47.0° N to 55.5° N
```

### Time Period
```python
YEARS: [2020, 2021, 2022, 2023]
```

### Thresholds
```python
WET_DAY_THRESHOLD: 10.0 mm/day
TEMP_HIGH_PERCENTILE: 90  (hot days)
TEMP_LOW_PERCENTILE: 10   (cold days)
PR_HIGH_PERCENTILE: 95    (wet days)
PR_LOW_PERCENTILE: 5      (dry days)
```

### Visualization
- **Projection**: Lambert Conformal (centered 10.5°E, 51°N)
- **Figure size**: 20-26 inches (landscape)
- **Resolution**: 300 DPI for publication quality

---

## Output Structure

```
outputs/
├── extremes/
│   ├── spatial_extremes_analysis/
│   │   └── wet_days_frequency_bias.png
│   │
│   └── compound_extremes/
│       ├── joint_distribution_differences.png
│       └── joint_distribution_summary.csv
```

---

## Key Variables Available After Execution

### spatial_extremes_analysis.ipynb
- `wet_days_era5`, `wet_days_pred10km`, `wet_days_mswx`: Spatial fields
- `wet_days_isimip_regridded`: ISIMIP regridded to MSWX grid
- `datasets`: List of (name, data, truth) tuples
- `bias_data`: Dictionary of bias fields by dataset

### compound_extremes_analysis.ipynb
- `joint_distributions`: Dictionary with model/reference/difference distributions
- `thresholds`: Percentile thresholds per dataset
- `hot_dry_events`, `hot_wet_events`, `cold_wet_events`: Compound event masks
- `summary_df`: Statistics summary table

---

## Usage Notes

1. **Consistency**: Both notebooks use the same 3 model-reference pairs
2. **Regridding**: ISIMIP requires xesmf; ensures grid compatibility
3. **Percentiles**: Computed independently per dataset (no cross-dataset normalization)
4. **Joint Distributions**: Based on spatial means for numerical stability
5. **Projections**: Lambert Conformal minimizes distortion for mid-latitude region

---

## Future Extensions

Potential analyses using the defined functions:
- Spatial mapping of all extreme indices (CDD, CWD, R10mm, R20mm, Rx1day, Rx5day)
- Temporal evolution of compound extreme frequencies
- Regional aggregation and trend analysis
- Extremal dependence metrics (tail correlations)
- Extremal indices correlations with compound events
