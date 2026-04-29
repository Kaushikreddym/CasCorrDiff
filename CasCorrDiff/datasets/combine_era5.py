import os
import glob
import re
import xarray as xr
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


# ERA5 variable code to CF-compliant name mapping
ERA5_VAR_MAPPING = {
    # Surface variables (sf/)
    "167": "tas",              # 2-meter temperature
    "168": "2d",              # 2-meter dewpoint temperature  
    "169": "rsds",              # Land-sea mask
    "201": "tasmax",             # Maximum 2m temperature
    "202": "tasmin",             # Minimum 2m temperature
    "228": "pr",               # Total precipitation
    
    # Pressure level variables (pl/)
    "129": "zg",                # Geopotential
    "130": "ta",                # Temperature
    "131": "ua",                # U-component of wind
    "132": "va",                # V-component of wind
    "133": "hus",                # Specific humidity
}


def get_cf_varname(era5_code, plev=None):
    """
    Convert ERA5 variable code to CF-compliant name.
    
    Parameters
    ----------
    era5_code : str
        ERA5 variable code (e.g., '167', '129')
    plev : float, optional
        Pressure level in Pa (e.g., 85000)
        
    Returns
    -------
    str
        CF-compliant variable name (e.g., 't2m', 'z850')
    """
    base_name = ERA5_VAR_MAPPING.get(era5_code, f"var{era5_code}")
    
    if plev is not None:
        # Convert Pa to hPa for naming (e.g., 85000 Pa -> 850 hPa)
        plev_hpa = int(plev / 100)
        return f"{base_name}{plev_hpa}"
    
    return base_name


def extract_dates(root="/beegfs/muduchuru/data/era5"):
    """Extract all available dates for each variable and return their intersection."""
    pattern = re.compile(r"_(\d{4}-\d{2})_")  # Match YYYY-MM format

    all_dates_per_var = []
    for sub in [os.path.join(root, "sf", "*"), os.path.join(root, "pl", "*")]:
        for var_dir in glob.glob(sub):
            files = glob.glob(os.path.join(var_dir, "*.nc"))
            dates = set()
            for f in files:
                m = pattern.search(os.path.basename(f))
                if m:
                    dates.add(m.group(1))
            if dates:
                all_dates_per_var.append(dates)

    common_dates = sorted(set.intersection(*all_dates_per_var))
    print(f"✅ Found {len(common_dates)} common dates (YYYY-MM) across all variables.")
    return common_dates


def decode_time_coordinate(time_values):
    """
    Decode time coordinate from various formats.
    Handles both standard datetime and day-as-decimal format (YYYYMMDD.f)
    """
    # Check if already datetime
    if hasattr(time_values, 'dtype') and 'datetime' in str(time_values.dtype):
        return pd.to_datetime(time_values)
    
    if str(time_values.dtype).startswith("float"):
        # Format: YYYYMMDD.f
        base_dates = pd.to_datetime(time_values.astype(int).astype(str), format="%Y%m%d")
        return base_dates
    else:
        # Try to convert to datetime
        return pd.to_datetime(time_values)


def combine_era5_channels_for_day(month_str, time_index, root="/beegfs/muduchuru/data/era5"):
    """
    Combine all ERA5 variables for a specific day from monthly files.
    Renames variables to CF-compliant names.
    
    Parameters
    ----------
    month_str : str
        Month in YYYY-MM format
    time_index : int or datetime
        Time index or datetime to extract from monthly files
    root : str
        Root directory containing sf/ and pl/ subdirectories
        
    Returns
    -------
    tuple
        (date_str, xr.Dataset) - Date string and combined dataset for that day
    """
    subdirs = [os.path.join(root, "sf", "*"), os.path.join(root, "pl", "*")]
    files = []
    for sub in subdirs:
        files.extend(glob.glob(os.path.join(sub, f"*_{month_str}_*.nc")))

    if not files:
        raise FileNotFoundError(f"No ERA5 files found for {month_str}")

    combined_vars = {}
    channel_names = []
    date_str = None

    for f in sorted(files):
        try:
            # Try to open with automatic time decoding first
            ds = xr.open_dataset(f)
            
            # Check if time needs manual decoding (for YYYYMMDD.f format)
            if "time" in ds.coords and 'units' in ds.time.attrs:
                units = ds.time.attrs.get('units', '')
                # If it's the YYYYMMDD.f format, we need to decode manually
                if 'day as %Y%m%d' in units:
                    ds.close()
                    ds = xr.open_dataset(f, decode_times=False)
                    time_decoded = decode_time_coordinate(ds.time.values)
                    ds = ds.assign_coords(time=("time", time_decoded))
            
            # Extract ERA5 variable code from filename
            basename = os.path.basename(f)
            era5_code = basename.split('_')[-1].replace('.nc', '')
            
            # Get original variable name from file
            orig_var = list(ds.data_vars.keys())[0]
            da = ds[orig_var]
            
            # Select the specific time index
            if "time" in da.dims:
                da = da.isel(time=time_index)
                # Get the date string from this time value
                if date_str is None:
                    date_str = pd.to_datetime(da.time.values).strftime("%Y-%m-%d")
                # Drop the time coordinate to avoid conflicts when combining
                da = da.drop_vars("time")
            
            # Check if this is a pressure level variable
            if "plev" in da.dims:
                # Split into separate variables for each level
                for plev_val in da.plev.values:
                    cf_name = get_cf_varname(era5_code, plev_val)
                    da_level = da.sel(plev=plev_val, drop=True)
                    combined_vars[cf_name] = da_level
                    channel_names.append(cf_name)
            else:
                # Surface variable
                cf_name = get_cf_varname(era5_code)
                combined_vars[cf_name] = da
                channel_names.append(cf_name)
                
        except Exception as e:
            print(f"⚠️ Skipping {f} for time {time_index}: {e}")

    if not combined_vars:
        raise ValueError(f"No valid variables found for {month_str} time {time_index}")

    # Stack all variables into a single array with channel dimension
    # Create a list of DataArrays with consistent coordinates
    data_arrays = []
    for cf_name in sorted(combined_vars.keys()):
        da = combined_vars[cf_name]
        # Expand dims to add channel dimension
        da_expanded = da.expand_dims(dim={'channel': [cf_name]})
        data_arrays.append(da_expanded)
    
    # Concatenate along channel dimension
    image = xr.concat(data_arrays, dim='channel')
    
    # Add time dimension
    time_coord = pd.to_datetime(date_str)
    image = image.expand_dims(dim={'time': [time_coord]})
    
    # Transpose to standard order: (time, channel, lat, lon)
    image = image.transpose('time', 'channel', 'lat', 'lon')
    
    # Create dataset with single 'image' variable
    ds_out = xr.Dataset({'image': image})
    
    # Sort by latitude (descending for standard orientation)
    if "lat" in ds_out.coords:
        ds_out = ds_out.sortby("lat", ascending=False)
    
    # Add metadata
    channel_names = sorted(combined_vars.keys())
    ds_out.attrs["variables"] = ",".join(channel_names)
    ds_out.attrs["source"] = "ERA5 reanalysis"
    ds_out.attrs["institution"] = "European Centre for Medium-Range Weather Forecasts"
    ds_out.attrs["Conventions"] = "CF-1.6"
    ds_out.attrs["date"] = date_str
    
    # Add metadata to image variable
    ds_out['image'].attrs["variables"] = ",".join(channel_names)

    return date_str, ds_out


def process_single_month(month_str, root, out_dir):
    """
    Process all days in a single month and save combined files.
    
    Parameters
    ----------
    month_str : str
        Month in YYYY-MM format
    root : str
        Root directory containing sf/ and pl/ subdirectories
    out_dir : str
        Output directory for combined files
        
    Returns
    -------
    tuple
        (month_str, days_processed, error_count)
    """
    days_processed = 0
    error_count = 0
    
    try:
        # Open one file to get the number of time steps
        sample_files = glob.glob(os.path.join(root, "sf", "*", f"*_{month_str}_*.nc"))
        if not sample_files:
            return (month_str, 0, 0, "No files found")
            
        with xr.open_dataset(sample_files[0], decode_times=False) as ds_sample:
            n_times = len(ds_sample.time)
        
        # Process each day in this month
        for time_idx in range(n_times):
            try:
                date_str, ds_out = combine_era5_channels_for_day(month_str, time_idx, root)
                nvars = len(ds_out.channel)  # Count channels, not data_vars
                
                out_filename = f"ERA5_{date_str}_{nvars}var.nc"
                out_path = os.path.join(out_dir, out_filename)

                # Save with compression
                encoding = {
                    'image': {"zlib": True, "complevel": 4},
                    'time': {"dtype": "int64"}
                }
                ds_out.to_netcdf(out_path, encoding=encoding)
                ds_out.close()
                days_processed += 1
                
            except Exception as e:
                error_count += 1
                print(f"  ❌ {month_str} day {time_idx}: {e}")
                
    except Exception as e:
        return (month_str, days_processed, error_count, str(e))
    
    return (month_str, days_processed, error_count, None)


def combine_all_common_dates(root="/beegfs/muduchuru/data/era5", out_dir=None, n_workers=10):
    """
    Loop through all common month files, extract daily data in parallel, and save combined files 
    as ERA5_<YYYY-MM-DD>.nc with CF-compliant variable names.
    
    Parameters
    ----------
    root : str
        Root directory containing sf/ and pl/ subdirectories
    out_dir : str, optional
        Output directory for combined files (default: root/combined)
    n_workers : int
        Number of parallel workers (default: 10)
    """
    if out_dir is None:
        out_dir = os.path.join(root, "combined")
    os.makedirs(out_dir, exist_ok=True)

    common_months = extract_dates(root)
    
    print(f"\n🚀 Starting parallel processing with {n_workers} workers...\n")
    
    total_days_processed = 0
    total_errors = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all months for parallel processing
        futures = {
            executor.submit(process_single_month, month, root, out_dir): month 
            for month in common_months
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            month = futures[future]
            try:
                month_str, days, errors, error_msg = future.result()
                total_days_processed += days
                total_errors += errors
                
                if error_msg and days == 0:
                    print(f"❌ {month_str}: {error_msg}")
                else:
                    status = "✅" if errors == 0 else "⚠️"
                    print(f"{status} {month_str}: {days} days processed, {errors} errors")
                    
            except Exception as e:
                print(f"❌ {month}: Unexpected error: {e}")
                total_errors += 1
    
    print(f"\n🎉 Finished! Processed {total_days_processed} days total with {total_errors} errors.")


if __name__ == "__main__":
    # === Run ===
    combine_all_common_dates(
        root="/data01/FDS/muduchuru/Atmos/ERA5/cmip_var/era5/daily_nc/europe",
        n_workers=20  # Adjust based on available CPU cores
    )