"""
Calculate normalization statistics for MSWX and HYRAS datasets.

Computes mean/std statistics with log transformation for precipitation channels.
Also computes quantile transforms for precipitation using sklearn's QuantileTransformer.
Supports both full extent and Germany spatial domain.

Usage:
    python calc_stats_mswxdwd.py --domain full
    python calc_stats_mswxdwd.py --domain germany
"""

import os
import sys
import glob
import json
import argparse
import pickle
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from sklearn.preprocessing import QuantileTransformer

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_valid_spatial_bounds(data_2d: np.ndarray):
    """
    Find rows and columns that contain at least one non-NaN value.
    
    Parameters
    ----------
    data_2d : np.ndarray (H, W)
        2D array with NaN values
        
    Returns
    -------
    row_slice, col_slice : tuple of slices
    """
    valid_rows = ~np.all(np.isnan(data_2d), axis=1)
    valid_cols = ~np.all(np.isnan(data_2d), axis=0)
    
    row_indices = np.where(valid_rows)[0]
    col_indices = np.where(valid_cols)[0]
    
    if len(row_indices) == 0 or len(col_indices) == 0:
        raise ValueError("All data is NaN - cannot find valid spatial bounds")
    
    row_slice = slice(row_indices[0], row_indices[-1] + 1)
    col_slice = slice(col_indices[0], col_indices[-1] + 1)
    
    return row_slice, col_slice


def get_spatial_extent(dwd_file: str):
    """Get spatial extent from a DWD file."""
    ds = xr.open_dataset(dwd_file)
    lat_min, lat_max = float(np.min(ds.lat)), float(np.max(ds.lat))
    lon_min, lon_max = float(np.min(ds.lon)), float(np.max(ds.lon))
    return (lat_min, lat_max, lon_min, lon_max)


def process_single_file(
    file_path: str,
    variable_name: str,
    spatial_slice: tuple,
    spatial_bounds: dict,
    use_log: bool,
    is_hyras: bool
):
    """
    Process a single file and return statistics (sum, sum_sq, count, min, max).
    Worker function for parallel processing - memory efficient.
    """
    try:
        with xr.open_dataset(file_path) as ds:
            # Get the data
            if variable_name not in ds.data_vars:
                return (0.0, 0.0, 0, np.inf, -np.inf)
            
            data_var = ds[variable_name]
            
            # Apply spatial bounds if provided (for both MSWX and HYRAS)
            if spatial_bounds is not None:
                if 'lat' in ds.coords and 'lon' in ds.coords:
                    lat = ds['lat']
                    lon = ds['lon']
                    
                    # Handle both 1D (MSWX) and 2D (HYRAS) lat/lon coords
                    if lat.ndim == 1:  # MSWX format
                        # Check if lat is descending (common in MSWX)
                        if lat.values[0] > lat.values[-1]:
                            # Descending: use reversed slice
                            data_var = data_var.sel(
                                lat=slice(spatial_bounds['lat_max'], spatial_bounds['lat_min']),
                                lon=slice(spatial_bounds['lon_min'], spatial_bounds['lon_max'])
                            )
                        else:
                            # Ascending: use normal slice
                            data_var = data_var.sel(
                                lat=slice(spatial_bounds['lat_min'], spatial_bounds['lat_max']),
                                lon=slice(spatial_bounds['lon_min'], spatial_bounds['lon_max'])
                            )
                    else:  # HYRAS format (2D lat/lon)
                        mask = (
                            (lat >= spatial_bounds['lat_min']) & 
                            (lat <= spatial_bounds['lat_max']) &
                            (lon >= spatial_bounds['lon_min']) & 
                            (lon <= spatial_bounds['lon_max'])
                        )
                        data_var = data_var.where(mask)
            
            data = data_var.values
            
            # Handle fill values
            fill_value = data_var.attrs.get('_FillValue', None)
            if fill_value is not None:
                data = np.where(data == fill_value, np.nan, data)
            
            # Initialize accumulators
            total_sum = 0.0
            total_sqsum = 0.0
            total_count = 0
            total_min = np.inf
            total_max = -np.inf
            
            # HYRAS has time dimension, MSWX has single timestep
            if is_hyras and data.ndim == 3:  # (time, y, x)
                # Process all timesteps
                for t in range(data.shape[0]):
                    time_slice = data[t]
                    
                    # Apply spatial cropping if provided
                    if spatial_slice is not None:
                        row_slice, col_slice = spatial_slice
                        time_slice = time_slice[row_slice, col_slice]
                    
                    # Get only valid (non-NaN) pixels
                    valid_mask = ~np.isnan(time_slice)
                    if use_log:
                        # For log transform, also exclude zeros and negatives
                        valid_mask = valid_mask & (time_slice > 0)
                    
                    valid_data = time_slice[valid_mask]
                    
                    if len(valid_data) > 0:
                        # Apply log transformation if needed
                        if use_log:
                            valid_data = np.log1p(valid_data)
                        
                        # Accumulate statistics
                        total_sum += np.sum(valid_data)
                        total_sqsum += np.sum(valid_data ** 2)
                        total_count += len(valid_data)
                        total_min = min(total_min, np.min(valid_data))
                        total_max = max(total_max, np.max(valid_data))
            else:  # MSWX single timestep
                # Remove time dimension if present
                if data.ndim == 3:
                    data = data[0]  # (time=1, lat, lon) -> (lat, lon)
                
                # Apply spatial cropping if provided
                if spatial_slice is not None:
                    row_slice, col_slice = spatial_slice
                    data = data[row_slice, col_slice]
                
                # Get only valid pixels
                valid_mask = ~np.isnan(data)
                if use_log:
                    valid_mask = valid_mask & (data > 0)
                
                valid_data = data[valid_mask]
                
                if len(valid_data) > 0:
                    # Apply log transformation if needed
                    if use_log:
                        valid_data = np.log1p(valid_data)
                    
                    # Accumulate statistics
                    total_sum += np.sum(valid_data)
                    total_sqsum += np.sum(valid_data ** 2)
                    total_count += len(valid_data)
                    total_min = min(total_min, np.min(valid_data))
                    total_max = max(total_max, np.max(valid_data))
            
            return (total_sum, total_sqsum, total_count, total_min, total_max)
                
    except Exception as e:
        return (0.0, 0.0, 0, np.inf, -np.inf)


def compute_channel_stats(
    files: list,
    channel_name: str,
    variable_name: str,
    spatial_slice: tuple = None,
    spatial_bounds: dict = None,
    use_log: bool = False,
    max_files: int = None,
    is_hyras: bool = False,
    n_workers: int = 8
):
    """
    Compute mean/std statistics for a single channel.
    
    Parameters
    ----------
    files : list
        List of netCDF file paths
    channel_name : str
        Name of the channel (subfolder name)
    variable_name : str
        Actual variable name in the NetCDF file
    spatial_slice : tuple, optional
        (row_slice, col_slice) for cropping
    spatial_bounds : dict, optional
        Lat/lon bounds: {'lat_min': float, 'lat_max': float, 'lon_min': float, 'lon_max': float}
    use_log : bool
        Whether to apply log transformation
    max_files : int, optional
        Maximum number of files to process (for testing)
    is_hyras : bool
        Whether this is HYRAS data (yearly files with multiple timesteps)
    n_workers : int
        Number of parallel workers (default: 8)
        
    Returns
    -------
    dict
        Statistics dictionary with mean, std, min, max
    """
    files_to_process = files[:max_files] if max_files else files
    
    print(f"  Computing stats for '{channel_name}' (variable: '{variable_name}') from {len(files_to_process)} files...")
    print(f"  Log transform: {use_log}, HYRAS format: {is_hyras}, Workers: {n_workers}")
    
    # Create partial function with fixed parameters
    process_func = partial(
        process_single_file,
        variable_name=variable_name,
        spatial_slice=spatial_slice,
        spatial_bounds=spatial_bounds,
        use_log=use_log,
        is_hyras=is_hyras
    )
    
    # Initialize accumulators for combining statistics
    total_sum = 0.0
    total_sqsum = 0.0
    total_count = 0
    global_min = np.inf
    global_max = -np.inf
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_func, f): f for f in files_to_process}
        
        with tqdm(total=len(files_to_process), desc=f"  {channel_name}") as pbar:
            for future in as_completed(futures):
                s, sq, c, mn, mx = future.result()
                total_sum += s
                total_sqsum += sq
                total_count += c
                if mn < global_min:
                    global_min = mn
                if mx > global_max:
                    global_max = mx
                pbar.update(1)
    
    if total_count == 0:
        print(f"    WARNING: No valid data found for {channel_name}")
        return {
            "mean": 0.0,
            "std": 1.0,
            "min": 0.0,
            "max": 1.0,
            "count": 0,
            "log_transformed": use_log
        }
    
    # Compute statistics from accumulated values
    mean = total_sum / total_count
    variance = (total_sqsum / total_count) - (mean ** 2)
    std = np.sqrt(max(0.0, variance))  # Avoid negative due to numerical errors
    
    stats = {
        "mean": float(mean),
        "std": float(std),
        "min": float(global_min),
        "max": float(global_max),
        "count": int(total_count),
        "log_transformed": use_log
    }
    
    print(f"    Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}, Count: {stats['count']:,}")
    
    return stats


def compute_quantile_transform(
    files: list,
    channel_name: str,
    variable_name: str,
    spatial_slice: tuple = None,
    spatial_bounds: dict = None,
    max_files: int = None,
    is_hyras: bool = False,
    n_workers: int = 8,
    n_quantiles: int = 1000
):
    """
    Compute quantile transform for a channel using sklearn's QuantileTransformer.
    This is useful for precipitation data to map to uniform distribution.
    
    Parameters
    ----------
    files : list
        List of netCDF file paths
    channel_name : str
        Name of the channel
    variable_name : str
        Actual variable name in the NetCDF file
    spatial_slice : tuple, optional
        (row_slice, col_slice) for cropping
    spatial_bounds : dict, optional
        Lat/lon bounds
    max_files : int, optional
        Maximum number of files to process
    is_hyras : bool
        Whether this is HYRAS data
    n_workers : int
        Number of parallel workers
    n_quantiles : int
        Number of quantile points (default: 1000)
        
    Returns
    -------
    dict
        Dictionary containing the fitted QuantileTransformer and metadata
    """
    files_to_process = files[:max_files] if max_files else files
    
    print(f"  Computing quantile transform for '{channel_name}' from {len(files_to_process)} files...")
    print(f"    HYRAS format: {is_hyras}, Workers: {n_workers}, N-Quantiles: {n_quantiles}")
    
    # Define worker function for collecting data
    def collect_valid_data(file_path: str):
        """Collect all valid (non-NaN, positive) data from a file."""
        try:
            with xr.open_dataset(file_path) as ds:
                if variable_name not in ds.data_vars:
                    return np.array([])
                
                data_var = ds[variable_name]
                
                # Apply spatial bounds
                if spatial_bounds is not None and 'lat' in ds.coords and 'lon' in ds.coords:
                    lat = ds['lat']
                    lon = ds['lon']
                    
                    if lat.ndim == 1:  # MSWX format
                        if lat.values[0] > lat.values[-1]:
                            data_var = data_var.sel(
                                lat=slice(spatial_bounds['lat_max'], spatial_bounds['lat_min']),
                                lon=slice(spatial_bounds['lon_min'], spatial_bounds['lon_max'])
                            )
                        else:
                            data_var = data_var.sel(
                                lat=slice(spatial_bounds['lat_min'], spatial_bounds['lat_max']),
                                lon=slice(spatial_bounds['lon_min'], spatial_bounds['lon_max'])
                            )
                    else:  # HYRAS format
                        mask = (
                            (lat >= spatial_bounds['lat_min']) & 
                            (lat <= spatial_bounds['lat_max']) &
                            (lon >= spatial_bounds['lon_min']) & 
                            (lon <= spatial_bounds['lon_max'])
                        )
                        data_var = data_var.where(mask)
                
                data = data_var.values
                
                # Handle fill values
                fill_value = data_var.attrs.get('_FillValue', None)
                if fill_value is not None:
                    data = np.where(data == fill_value, np.nan, data)
                
                # Collect all valid samples
                valid_data = []
                
                if is_hyras and data.ndim == 3:  # (time, y, x)
                    for t in range(data.shape[0]):
                        time_slice = data[t]
                        if spatial_slice is not None:
                            row_slice, col_slice = spatial_slice
                            time_slice = time_slice[row_slice, col_slice]
                        
                        # Get valid pixels (non-NaN, positive)
                        valid_mask = ~np.isnan(time_slice) & (time_slice > 0)
                        valid_data.extend(time_slice[valid_mask].flatten().tolist())
                else:  # MSWX single timestep
                    if data.ndim == 3:
                        data = data[0]
                    
                    if spatial_slice is not None:
                        row_slice, col_slice = spatial_slice
                        data = data[row_slice, col_slice]
                    
                    valid_mask = ~np.isnan(data) & (data > 0)
                    valid_data.extend(data[valid_mask].flatten().tolist())
                
                return np.array(valid_data, dtype=np.float32)
        
        except Exception as e:
            return np.array([])
    
    # Collect data in parallel with progress bar
    all_data = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(collect_valid_data, f): f for f in files_to_process}
        
        with tqdm(total=len(files_to_process), desc=f"  {channel_name} (collecting)") as pbar:
            for future in as_completed(futures):
                data = future.result()
                if len(data) > 0:
                    all_data.extend(data.tolist())
                pbar.update(1)
    
    if len(all_data) == 0:
        print(f"    WARNING: No valid data found for {channel_name}")
        return {
            "transformer": None,
            "count": 0,
            "error": "No valid data"
        }
    
    # Reshape data for sklearn (needs 2D array: n_samples x n_features)
    all_data = np.array(all_data, dtype=np.float32).reshape(-1, 1)
    
    print(f"    Collected {len(all_data):,} samples")
    print(f"    Min: {all_data.min():.6f}, Median: {np.median(all_data):.6f}, Max: {all_data.max():.6f}")
    
    # Fit QuantileTransformer
    print(f"    Fitting QuantileTransformer with {n_quantiles} quantiles...")
    qt = QuantileTransformer(
        n_quantiles=min(n_quantiles, len(all_data)),
        output_distribution='normal',
        subsample=int(1e8),  # Use up to 100M samples
        random_state=42
    )
    qt.fit(all_data)
    
    result = {
        "transformer": qt,
        "count": len(all_data),
        "min": float(all_data.min()),
        "max": float(all_data.max()),
        "median": float(np.median(all_data)),
        "n_quantiles": min(n_quantiles, len(all_data))
    }
    
    print(f"    ✅ QuantileTransformer fitted successfully")
    print(f"    Min: {result['min']:.6f}, Median: {result['median']:.6f}, Max: {result['max']:.6f}")
    
    return result


def compute_dwd_stats(
    data_path: str,
    output_file: str,
    spatial_bounds: dict = None,
    max_files: int = None,
    n_workers: int = 8
):
    """
    Compute statistics for DWD/HYRAS high-resolution data.
    
    Parameters
    ----------
    data_path : str
        Root data directory (e.g., /beegfs/muduchuru/data/hyras)
    output_file : str
        Output JSON file path
    max_files : int, optional
        Maximum files per channel (for testing)
    """
    print("\n" + "="*60)
    print("Computing HYRAS (Target/High-Res) Statistics")
    print("="*60)
    
    # Mapping from subfolder name to variable name in NetCDF
    HYRAS_CHANNEL_MAP = {
        "pr": "pr",
        "tas": "tas",
        "tasmax": "tasmax",
        "tasmin": "tasmin",
        "hurs": "hurs",
        "rsds": "rsds"
    }
    
    stats_dwd = {}
    
    # Get list of available channels
    channels = [d for d in os.listdir(data_path) 
                if os.path.isdir(os.path.join(data_path, d)) and d in HYRAS_CHANNEL_MAP]
    
    if not channels:
        print(f"ERROR: No valid HYRAS channels found in {data_path}")
        return
    
    print(f"Found channels: {', '.join(channels)}\n")
    
    # Spatial slice not needed when using spatial_bounds filtering
    row_slice = None
    col_slice = None
    
    if spatial_bounds:
        print(f"Using spatial bounds: lat [{spatial_bounds['lat_min']}, {spatial_bounds['lat_max']}], "
              f"lon [{spatial_bounds['lon_min']}, {spatial_bounds['lon_max']}]\n")
    else:
        print("Processing full spatial extent\n")
    
    # Process each channel
    for ch in channels:
        var_name = HYRAS_CHANNEL_MAP[ch]
        files = sorted(glob.glob(os.path.join(data_path, ch, "*.nc")))
        
        if len(files) == 0:
            print(f"  WARNING: No files found for channel '{ch}'")
            continue
        
        # Determine if this channel needs log transformation
        use_log = "pr" in ch.lower() or "precip" in ch.lower()
        
        stats_dwd[ch.lower()] = compute_channel_stats(
            files=files,
            channel_name=ch,
            variable_name=var_name,
            spatial_slice=(row_slice, col_slice) if row_slice else None,
            spatial_bounds=spatial_bounds,
            use_log=use_log,
            max_files=max_files,
            is_hyras=True,
            n_workers=n_workers
        )
    
    # Save statistics
    with open(output_file, "w") as f:
        json.dump(stats_dwd, f, indent=2)
    
    print(f"\n✅ HYRAS statistics saved to: {output_file}")
    print(f"   Channels processed: {', '.join(stats_dwd.keys())}")
    
    # Compute quantile transforms for precipitation channel
    print(f"\n" + "="*60)
    print("Computing Quantile Transforms for HYRAS Precipitation")
    print("="*60)
    
    if "pr" in stats_dwd:
        pr_files = sorted(glob.glob(os.path.join(data_path, "pr", "*.nc")))
        if pr_files:
            quantile_data = compute_quantile_transform(
                files=pr_files,
                channel_name="pr",
                variable_name="pr",
                spatial_slice=(row_slice, col_slice) if row_slice else None,
                spatial_bounds=spatial_bounds,
                max_files=max_files,
                is_hyras=True,
                n_workers=n_workers
            )
            
            # Save quantile transform as pickle file
            quantile_file = output_file.replace("_log.json", "_quantile_transform.pkl")
            with open(quantile_file, "wb") as f:
                pickle.dump(quantile_data, f)
            print(f"\n✅ HYRAS quantile transform saved to: {quantile_file}")
        else:
            print("WARNING: No precipitation files found for quantile transform")
    else:
        print("WARNING: No precipitation statistics found")


def compute_mswx_stats(
    data_path: str,
    output_file: str,
    spatial_bounds: dict = None,
    max_files: int = None,
    n_workers: int = 8
):
    """
    Compute statistics for MSWX low-resolution data.
    
    Parameters
    ----------
    data_path : str
        Root data directory (e.g., /beegfs/muduchuru/data/mswx)
    output_file : str
        Output JSON file path
    spatial_bounds : dict, optional
        Lat/lon bounds for Germany domain
    max_files : int, optional
        Maximum files per channel (for testing)
    n_workers : int
        Number of parallel workers (default: 8)
    """
    print("\n" + "="*60)
    print("Computing MSWX (Input/Low-Res) Statistics")
    print("="*60)
    
    if spatial_bounds:
        print(f"Using spatial bounds: lat [{spatial_bounds['lat_min']}, {spatial_bounds['lat_max']}], "
              f"lon [{spatial_bounds['lon_min']}, {spatial_bounds['lon_max']}]\n")
    else:
        print("Processing full spatial extent\n")
    
    # Mapping from subfolder name to variable name in NetCDF
    MSWX_CHANNEL_MAP = {
        "pr": "precipitation",
        "tas": "air_temperature",
        "tasmax": "air_temperature",
        "tasmin": "air_temperature",
        "hurs": "relative_humidity",
        "rsds": "downward_shortwave_radiation"
    }
    
    stats_mswx = {}
    
    # Get list of available channels
    channels = [d for d in os.listdir(data_path) 
                if os.path.isdir(os.path.join(data_path, d)) and d in MSWX_CHANNEL_MAP]
    
    if not channels:
        print(f"ERROR: No valid MSWX channels found in {data_path}")
        return
    
    print(f"Found channels: {', '.join(channels)}\n")
    
    # Process each channel
    for ch in channels:
        var_name = MSWX_CHANNEL_MAP[ch]
        files = sorted(glob.glob(os.path.join(data_path, ch, "*.nc")))
        
        if len(files) == 0:
            print(f"  WARNING: No files found for channel '{ch}'")
            continue
        
        # Determine if this channel needs log transformation
        use_log = "pr" in ch.lower() or "precip" in ch.lower()
        
        # Apply spatial bounds if provided, otherwise use global data
        stats_mswx[ch] = compute_channel_stats(
            files=files,
            channel_name=ch,
            variable_name=var_name,
            spatial_slice=None,
            spatial_bounds=spatial_bounds,
            use_log=use_log,
            max_files=max_files,
            is_hyras=False,
            n_workers=n_workers
        )
    
    # Save statistics
    with open(output_file, "w") as f:
        json.dump(stats_mswx, f, indent=2)
    
    print(f"\n✅ MSWX statistics saved to: {output_file}")
    print(f"   Channels processed: {', '.join(stats_mswx.keys())}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics for MSWX and HYRAS datasets"
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["germany", "full"],
        default="full",
        help="Spatial domain: germany (47-55°N, 6-15°E) or full (default: full)"
    )
    parser.add_argument(
        "--mswx-path",
        type=str,
        default="/beegfs/muduchuru/data/mswx",
        help="MSWX data directory (default: /beegfs/muduchuru/data/mswx)"
    )
    parser.add_argument(
        "--hyras-path",
        type=str,
        default="/beegfs/muduchuru/data/HYRAS_DAILY",
        help="HYRAS data directory (default: /beegfs/muduchuru/data/hyras)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/beegfs/muduchuru/data",
        help="Output directory for stats files (default: /beegfs/muduchuru/data)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum files per channel for testing (default: all files)"
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=80,
        help="Number of parallel workers (default: 80)"
    )
    parser.add_argument(
        "--skip-mswx",
        action="store_true",
        help="Skip MSWX statistics computation"
    )
    parser.add_argument(
        "--skip-hyras",
        action="store_true",
        help="Skip HYRAS statistics computation"
    )
    parser.add_argument(
        "--compute-quantile-transforms",
        action="store_true",
        help="Compute quantile transforms for precipitation (saves as .pkl files)"
    )
    parser.add_argument(
        "--n-quantiles",
        type=int,
        default=1000,
        help="Number of quantile points for quantile transform (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define spatial bounds for Germany domain
    spatial_bounds = None
    if args.domain == "germany":
        spatial_bounds = {
            'lat_min': 47.0,
            'lat_max': 55.0,
            'lon_min': 6.0,
            'lon_max': 15.0
        }
    
    # Setup output filenames
    domain_suffix = "_germany" if args.domain == "germany" else ""
    output_hyras = os.path.join(args.output_dir, f"hyras_stats{domain_suffix}_log.json")
    output_mswx = os.path.join(args.output_dir, f"mswx_stats{domain_suffix}_log.json")
    
    print("\n" + "="*60)
    print(f"MSWX-HYRAS Statistics Calculation")
    print("="*60)
    print(f"Domain: {args.domain}")
    if spatial_bounds:
        print(f"Spatial bounds: lat [{spatial_bounds['lat_min']}, {spatial_bounds['lat_max']}], "
              f"lon [{spatial_bounds['lon_min']}, {spatial_bounds['lon_max']}]")
    print(f"MSWX path: {args.mswx_path}")
    print(f"HYRAS path: {args.hyras_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parallel workers: {args.n_workers}")
    if args.max_files:
        print(f"Max files per channel: {args.max_files}")
    if args.compute_quantile_transforms:
        print(f"Compute quantile transforms: YES (n_quantiles={args.n_quantiles})")
    print("="*60)
    
    # Compute HYRAS statistics
    if not args.skip_hyras:
        compute_dwd_stats(
            data_path=args.hyras_path,
            output_file=output_hyras,
            spatial_bounds=spatial_bounds,
            max_files=args.max_files,
            n_workers=args.n_workers
        )
    
    # Compute MSWX statistics
    if not args.skip_mswx:
        compute_mswx_stats(
            data_path=args.mswx_path,
            output_file=output_mswx,
            spatial_bounds=spatial_bounds,
            max_files=args.max_files,
            n_workers=args.n_workers
        )
    
    print("\n" + "="*60)
    print("🎉 Statistics computation complete!")
    print("="*60)
    if not args.skip_hyras:
        print(f"HYRAS stats: {output_hyras}")
    if not args.skip_mswx:
        print(f"MSWX stats: {output_mswx}")
    
    # Compute quantile transforms if requested
    if args.compute_quantile_transforms:
        print(f"\n" + "="*60)
        print("Computing Quantile Transforms")
        print("="*60)
        
        if not args.skip_hyras:
            hyras_qt_file = output_hyras.replace("_log.json", "_quantile_transform.pkl")
            pr_files = sorted(glob.glob(os.path.join(args.hyras_path, "pr", "*.nc")))
            if pr_files:
                print(f"\nProcessing HYRAS precipitation for quantile transform...")
                qt_data = compute_quantile_transform(
                    files=pr_files,
                    channel_name="pr",
                    variable_name="pr",
                    spatial_slice=None,
                    spatial_bounds=spatial_bounds,
                    max_files=args.max_files,
                    is_hyras=True,
                    n_workers=args.n_workers,
                    n_quantiles=args.n_quantiles
                )
                with open(hyras_qt_file, "wb") as f:
                    pickle.dump(qt_data, f)
                print(f"✅ HYRAS quantile transform saved to: {hyras_qt_file}")
        
        if not args.skip_mswx:
            mswx_qt_file = output_mswx.replace("_log.json", "_quantile_transform.pkl")
            pr_files = sorted(glob.glob(os.path.join(args.mswx_path, "pr", "*.nc")))
            if pr_files:
                print(f"\nProcessing MSWX precipitation for quantile transform...")
                qt_data = compute_quantile_transform(
                    files=pr_files,
                    channel_name="pr",
                    variable_name="precipitation",
                    spatial_slice=None,
                    spatial_bounds=spatial_bounds,
                    max_files=args.max_files,
                    is_hyras=False,
                    n_workers=args.n_workers,
                    n_quantiles=args.n_quantiles
                )
                with open(mswx_qt_file, "wb") as f:
                    pickle.dump(qt_data, f)
                print(f"✅ MSWX quantile transform saved to: {mswx_qt_file}")
    print("\nTo use these stats, update your dataset initialization:")
    if not args.skip_hyras:
        print(f"  stats_hyras='{output_hyras}'")
    if not args.skip_mswx:
        print(f"  stats_mswx='{output_mswx}'")
    print("="*60)


if __name__ == "__main__":
    main()
