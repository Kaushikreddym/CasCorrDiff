import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import dask
from dask.distributed import Client, LocalCluster
import dask.array as da

import glob
import os
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error, r2_score
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class ERA5MSWX:
    """Class for loading and processing 10km ERA5-MSWX model data"""
    
    def __init__(self, base_path="/beegfs/muduchuru/codes/physicsnemo/examples/weather/corrdiff/"):
        """
        Initialize ERA5MSWX data loader
        
        Parameters:
        -----------
        base_path : str
            Base path to the model data directory
        """
        self.base_path = base_path
        self.data_path = os.path.join(base_path, "generated/")
        self.input = None
        self.prediction = None
        self.truth = None
        self.inv = None
        self.years = None
        
    def load(self, years, chunks={"time": 10}):
        """
        Load ERA5MSWX data for specified years into xarray datasets
        
        Parameters:
        -----------
        years : list of int
            Years to load data for
        chunks : dict, optional
            Chunking specification for dask arrays
            
        Returns:
        --------
        self : ERA5MSWX
            Returns self for method chaining
        """
        print("Loading 10km ERA5-MSWX data...")
        
        # Find files matching requested years
        era5mswx_files = sorted([
            fp for fp in glob.glob(f"{self.data_path}era5mswx_corrdiff_fulldomain_*.nc")
            if int(os.path.basename(fp).split("_")[-1].split(".")[0]) in years
        ])
        
        found_years = [
            int(os.path.basename(f).split("_")[-1].split(".")[0])
            for f in era5mswx_files
        ]
        
        print(f"Found {len(era5mswx_files)} files for 10km model")
        print(f"Years: {sorted(found_years)}")
        
        if len(era5mswx_files) == 0:
            raise ValueError(f"No files found for years: {years}")
        
        # Load datasets with standard chunking
        self.input = xr.open_mfdataset(
            era5mswx_files,
            group="input",
            combine="nested",
            concat_dim="time",
            parallel=False,
            chunks=chunks
        )
        
        self.prediction = xr.open_mfdataset(
            era5mswx_files,
            group="prediction",
            combine="nested",
            concat_dim="time",
            parallel=False,
            chunks=chunks
        )
        
        self.truth = xr.open_mfdataset(
            era5mswx_files,
            group="truth",
            combine="nested",
            concat_dim="time",
            parallel=False,
            chunks=chunks
        )
        
        self.inv = xr.open_mfdataset(
            era5mswx_files,
            combine="nested",
            concat_dim="time",
            parallel=False,
            chunks=chunks,
            decode_times=True,
            use_cftime=False
        )
        
        # Use coordinates from inv dataset (ungrouped, complete coordinates)
        print("Assigning coordinates from inv dataset (time, lat, lon) to all datasets...")
        
        # Get time from inv dataset
        inv_time = pd.to_datetime(self.inv.time.values)
        
        # Access lat/lon from inv coordinates or data variables (select first timestep)
        if 'lat' in self.inv.coords:
            inv_lat = self.inv.coords['lat']
        elif 'lat' in self.inv.data_vars:
            inv_lat = self.inv['lat']
        else:
            print("  Warning: Could not find lat in inv dataset")
            inv_lat = None
        
        # Select first timestep if lat/lon have time dimension
        if inv_lat is not None and 'time' in inv_lat.dims:
            inv_lat = inv_lat.isel(time=0)
        
        if 'lon' in self.inv.coords:
            inv_lon = self.inv.coords['lon']
        elif 'lon' in self.inv.data_vars:
            inv_lon = self.inv['lon']
        else:
            print("  Warning: Could not find lon in inv dataset")
            inv_lon = None
        
        # Select first timestep if lat/lon have time dimension
        if inv_lon is not None and 'time' in inv_lon.dims:
            inv_lon = inv_lon.isel(time=0)
        
        print(f"  Inv time length: {len(inv_time)}")
        print(f"  Input time length: {len(self.input.time)}")
        print(f"  Prediction time length: {len(self.prediction.time)}")
        print(f"  Truth time length: {len(self.truth.time)}")
        
        # Align all to inv time dimension and coordinates
        coords_dict = {'time': inv_time}
        if inv_lat is not None:
            coords_dict['lat'] = inv_lat
        if inv_lon is not None:
            coords_dict['lon'] = inv_lon
        
        self.input = self.input.assign_coords(coords_dict)
        self.prediction = self.prediction.assign_coords(coords_dict)
        self.truth = self.truth.assign_coords(coords_dict)
        
        # Convert ERA5 input precipitation from m/day to mm/day
        if 'pr' in self.input.data_vars:
            print("Converting ERA5 input precipitation from m/day to mm/day...")
            self.input['pr'] = self.input['pr'] * 1000.0
        
        # Convert ERA5 input temperature from Kelvin to Celsius
        temp_vars = ['tas', 'tasmin', 'tasmax']
        for var in temp_vars:
            if var in self.input.data_vars:
                print(f"Converting ERA5 input {var} from Kelvin to Celsius...")
                self.input[var] = self.input[var] - 273.15
        
        # Add units attributes to all groups
        # Input group
        for var in temp_vars:
            if var in self.input.data_vars:
                self.input[var].attrs['units'] = 'degC'
        if 'pr' in self.input.data_vars:
            self.input['pr'].attrs['units'] = 'mm/day'
        
        # Prediction group
        for var in temp_vars:
            if var in self.prediction.data_vars:
                self.prediction[var].attrs['units'] = 'degC'
        if 'pr' in self.prediction.data_vars:
            self.prediction['pr'].attrs['units'] = 'mm/day'
        
        # Truth group
        for var in temp_vars:
            if var in self.truth.data_vars:
                self.truth[var].attrs['units'] = 'degC'
        if 'pr' in self.truth.data_vars:
            self.truth['pr'].attrs['units'] = 'mm/day'
        
        self.years = sorted(found_years)
        
        print(f"10km data loaded - Time range: {self.inv.time.min().values} to {self.inv.time.max().values}")
        print(f"10km grid shape: {self.inv.lat.isel(time=0).shape}")
        
        return self
    
    def extract_GHCN(self, ds_obs, variable):
        """
        Extract variable data at GHCN station locations
        
        Parameters:
        -----------
        ds_obs : xarray.Dataset
            GHCN observations dataset (load using load_GHCN function)
        variable : str
            Variable name to extract (e.g., 'pr', 'tas', 'tasmin', 'tasmax')
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'observations': xarray.DataArray with GHCN observations for the variable
            - 'model_input': xarray.DataArray with model input at stations
            - 'model_prediction': xarray.DataArray with model predictions at stations
            - 'model_truth': xarray.DataArray with truth data at stations
            - 'indices': tuple of (y_indices, x_indices) for station locations
            - 'variable': name of the extracted variable
        """
        if self.input is None or self.prediction is None or self.inv is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        # Extract model data at stations
        print(f"\n=== Extracting {variable} from ERA5MSWX at GHCN stations ===")
        ds_obs_aligned, model_input_stations, model_prediction_stations, indices = \
            extract_model_at_stations(
                self.input, 
                self.prediction, 
                self.inv, 
                ds_obs, 
                "ERA5MSWX-10km"
            )
        
        # Extract truth data at stations
        truth_normalized = self.truth.assign_coords(time=pd.to_datetime(self.truth.time.values))
        obs_times = pd.to_datetime(ds_obs_aligned.time.values)
        common_times = obs_times.intersection(pd.to_datetime(self.inv.time.values))
        
        try:
            truth_aligned = truth_normalized.sel(time=common_times)
        except (KeyError, TypeError):
            truth_mask = truth_normalized.time.isin(common_times)
            truth_aligned = truth_normalized.where(truth_mask, drop=True)
        
        y_indices, x_indices = indices
        y_indices_da = xr.DataArray(y_indices, dims="station")
        x_indices_da = xr.DataArray(x_indices, dims="station")
        model_truth_stations = truth_aligned.isel(x=x_indices_da, y=y_indices_da)
        
        print(f"\n=== Extraction complete ===")
        print(f"Variable: {variable}")
        print(f"Stations: {len(ds_obs_aligned.station)}")
        print(f"Timesteps: {len(ds_obs_aligned.time)}")
        
        return {
            'observations': ds_obs_aligned[variable],
            'model_input': model_input_stations[variable],
            'model_prediction': model_prediction_stations[variable],
            'model_truth': model_truth_stations[variable],
            'indices': indices,
            'variable': variable
        }


class ISIMIP_ERA5:
    """Class for loading and processing ISIMIP BCSD-corrected ERA5 data"""
    
    def __init__(self, base_path="/beegfs/muduchuru/codes/diffusion_output_processing/bcsd/bcsd_outputs/", variable="pr"):
        """
        Initialize ISIMIP_ERA5 data loader
        
        Parameters:
        -----------
        base_path : str
            Base path to the BCSD output directory
        variable : str
            Variable name (e.g., 'pr', 'tasmax', 'tasmin')
        """
        self.base_path = base_path
        self.variable = variable
        self.prediction = None
        self.truth = None
        self.inv = None
        self.years = None
        
    def load(self, years, chunks={"time": 10}):
        """
        Load ISIMIP BCSD data for specified years into xarray datasets
        
        Parameters:
        -----------
        years : list of int
            Years to load data for
        chunks : dict, optional
            Chunking specification for dask arrays
            
        Returns:
        --------
        self : ISIMIP_ERA5
            Returns self for method chaining
        """
        print("\nLoading ISIMIP BCSD data...")
        
        # Find files matching requested years in variable-specific subdirectory
        search_path = os.path.join(self.base_path, self.variable, f"bcsd_{self.variable}_*.nc")
        isimip_files = sorted([
            fp for fp in glob.glob(search_path)
            if int(os.path.basename(fp).split("_")[-1].split(".")[0]) in years
        ])
        
        found_years = [
            int(os.path.basename(f).split("_")[-1].split(".")[0])
            for f in isimip_files
        ]
        
        print(f"Found {len(isimip_files)} files for ISIMIP BCSD")
        print(f"Years: {sorted(found_years)}")
        
        if len(isimip_files) == 0:
            raise ValueError(f"No files found for years: {years}")
        
        # Load datasets
        print("Loading BCSD predictions...")
        ds_list_pred = []
        ds_list_obs = []
        
        for fp in isimip_files:
            ds = xr.open_dataset(fp, chunks=chunks)
            
            # Extract prediction and truth using variable-specific names
            var_bcsd = f"{self.variable}_bcsd"
            var_obs = f"{self.variable}_obs"
            
            ds_pred = ds[[var_bcsd, 'lat', 'lon']]
            ds_obs = ds[[var_obs, 'lat', 'lon']]
            
            ds_list_pred.append(ds_pred)
            ds_list_obs.append(ds_obs)
        
        # Concatenate along time and rename to generic variable name
        self.prediction = xr.concat(ds_list_pred, dim='time').rename({var_bcsd: self.variable})
        self.truth = xr.concat(ds_list_obs, dim='time').rename({var_obs: self.variable})
        
        # Verify time dimensions match
        print(f"Prediction time length: {len(self.prediction.time)}")
        print(f"Truth time length: {len(self.truth.time)}")
        
        if len(self.prediction.time) != len(self.truth.time):
            print(f"Warning: Time dimension mismatch - aligning to prediction length")
            min_len = min(len(self.prediction.time), len(self.truth.time))
            self.prediction = self.prediction.isel(time=slice(0, min_len))
            self.truth = self.truth.isel(time=slice(0, min_len))
        
        # Drop NaN-only rows/columns after concatenation (more efficient)
        print("Dropping NaN-only lat/lon coordinates...")
        self.prediction = (self.prediction
                           .dropna(how='all', dim='lat')
                           .dropna(how='all', dim='lon'))
        
        self.truth = (self.truth
                      .dropna(how='all', dim='lat')
                      .dropna(how='all', dim='lon'))
        
        # Create inventory dataset (use prediction for grid)
        self.inv = xr.Dataset({
            'lat': self.prediction.lat,
            'lon': self.prediction.lon,
            'time': self.prediction.time
        })
        
        # Normalize time coordinates
        print("Normalizing time coordinates...")
        root_time = pd.to_datetime(self.inv.time.values)
        
        self.prediction = self.prediction.assign_coords(time=root_time)
        self.truth = self.truth.assign_coords(time=root_time)
        self.inv = self.inv.assign_coords(time=root_time)
        
        self.years = sorted(found_years)
        
        # Add units attributes based on variable type
        if self.variable == 'pr':
            # Precipitation
            if self.variable in self.prediction.data_vars:
                self.prediction[self.variable].attrs['units'] = 'mm/day'
            if self.variable in self.truth.data_vars:
                self.truth[self.variable].attrs['units'] = 'mm/day'
        elif self.variable in ['tasmin', 'tasmax', 'tas']:
            # Temperature
            if self.variable in self.prediction.data_vars:
                self.prediction[self.variable].attrs['units'] = 'degC'
            if self.variable in self.truth.data_vars:
                self.truth[self.variable].attrs['units'] = 'degC'
        
        print(f"ISIMIP data loaded - Time range: {self.inv.time.min().values} to {self.inv.time.max().values}")
        print(f"Grid shape: lat={len(self.inv.lat)}, lon={len(self.inv.lon)}")
        
        return self
    
    def extract_GHCN(self, ds_obs, variable):
        """
        Extract variable data at GHCN station locations
        
        Parameters:
        -----------
        ds_obs : xarray.Dataset
            GHCN observations dataset (load using load_GHCN function)
        variable : str
            Variable name to extract (e.g., 'pr')
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'observations': xarray.DataArray with GHCN observations for the variable
            - 'model_prediction': xarray.DataArray with BCSD predictions at stations
            - 'model_truth': xarray.DataArray with BCSD observations at stations
            - 'indices': tuple of (y_indices, x_indices) for station locations
            - 'variable': name of the extracted variable
        """
        if self.prediction is None or self.inv is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        # Extract model data at stations
        print(f"\n=== Extracting {variable} from ISIMIP BCSD at GHCN stations ===")
        ds_obs_aligned, _, model_prediction_stations, indices = \
            extract_model_at_stations(
                self.prediction,  # Use prediction as "input" (no separate input for BCSD)
                self.prediction,
                self.inv,
                ds_obs,
                "ISIMIP-BCSD"
            )
        
        # Extract truth data at stations
        truth_normalized = self.truth.assign_coords(time=pd.to_datetime(self.truth.time.values))
        obs_times = pd.to_datetime(ds_obs_aligned.time.values)
        common_times = obs_times.intersection(pd.to_datetime(self.inv.time.values))
        
        try:
            truth_aligned = truth_normalized.sel(time=common_times)
        except (KeyError, TypeError):
            truth_mask = truth_normalized.time.isin(common_times)
            truth_aligned = truth_normalized.where(truth_mask, drop=True)
        
        y_indices, x_indices = indices
        y_indices_da = xr.DataArray(y_indices, dims="station")
        x_indices_da = xr.DataArray(x_indices, dims="station")
        model_truth_stations = truth_aligned.isel(lat=y_indices_da, lon=x_indices_da)
        
        print(f"\n=== Extraction complete ===")
        print(f"Variable: {variable}")
        print(f"Stations: {len(ds_obs_aligned.station)}")
        print(f"Timesteps: {len(ds_obs_aligned.time)}")
        
        return {
            'observations': ds_obs_aligned[variable],
            'model_prediction': model_prediction_stations[variable],
            'model_truth': model_truth_stations[variable],
            'indices': indices,
            'variable': variable
        }


class MSWXDWD:
    """Class for loading and processing 1km MSWX-DWD model data"""
    
    def __init__(self, base_path="/beegfs/muduchuru/codes/physicsnemo/examples/weather/corrdiff/"):
        """
        Initialize MSWXDWD data loader
        
        Parameters:
        -----------
        base_path : str
            Base path to the model data directory
        """
        self.base_path = base_path
        self.data_path = os.path.join(base_path, "generated/combined/")
        self.input = None
        self.prediction = None
        self.truth = None
        self.inv = None
        self.years = None
        
    def load(self, years, chunks={"time": 10}):
        """
        Load MSWXDWD data for specified years into xarray datasets
        
        Parameters:
        -----------
        years : list of int
            Years to load data for
        chunks : dict, optional
            Chunking specification for dask arrays
            
        Returns:
        --------
        self : MSWXDWD
            Returns self for method chaining
        """
        print("\nLoading 1km MSWX-DWD data...")
        
        # Find files matching requested years
        mswxdwd_files = sorted([
            fp for fp in glob.glob(f"{self.data_path}mswxdwd_combined_*.nc")
            if int(os.path.basename(fp).split("_")[-1].split(".")[0]) in years
        ])
        
        found_years = [
            int(os.path.basename(f).split("_")[-1].split(".")[0])
            for f in mswxdwd_files
        ]
        
        print(f"Found {len(mswxdwd_files)} files for 1km model")
        print(f"Years: {sorted(found_years)}")
        
        if len(mswxdwd_files) == 0:
            raise ValueError(f"No files found for years: {years}")
        
        # Load datasets
        self.input = xr.open_mfdataset(
            mswxdwd_files,
            group="input",
            combine="nested",
            concat_dim="time",
            parallel=False,
            chunks=chunks
        )
        
        self.prediction = xr.open_mfdataset(
            mswxdwd_files,
            group="prediction",
            combine="nested",
            concat_dim="time",
            parallel=False,
            chunks=chunks
        )
        
        self.truth = xr.open_mfdataset(
            mswxdwd_files,
            group="truth",
            combine="nested",
            concat_dim="time",
            parallel=False,
            chunks=chunks
        )
        
        self.inv = xr.open_mfdataset(
            mswxdwd_files,
            combine="nested",
            concat_dim="time",
            parallel=False,
            chunks=chunks,
            decode_times=True,
            use_cftime=False
        )
        
        # Use coordinates from inv dataset (ungrouped, complete coordinates)
        print("Assigning coordinates from inv dataset (time, lat, lon) to all datasets...")
        
        # Get time from inv dataset
        inv_time = pd.to_datetime(self.inv.time.values)
        
        # Access lat/lon from inv coordinates or data variables (select first timestep)
        if 'lat' in self.inv.coords:
            inv_lat = self.inv.coords['lat']
        elif 'lat' in self.inv.data_vars:
            inv_lat = self.inv['lat']
        else:
            print("  Warning: Could not find lat in inv dataset")
            inv_lat = None
        
        # Select first timestep if lat/lon have time dimension
        if inv_lat is not None and 'time' in inv_lat.dims:
            inv_lat = inv_lat.isel(time=0)
        
        if 'lon' in self.inv.coords:
            inv_lon = self.inv.coords['lon']
        elif 'lon' in self.inv.data_vars:
            inv_lon = self.inv['lon']
        else:
            print("  Warning: Could not find lon in inv dataset")
            inv_lon = None
        
        # Select first timestep if lat/lon have time dimension
        if inv_lon is not None and 'time' in inv_lon.dims:
            inv_lon = inv_lon.isel(time=0)
        
        print(f"  Inv time length: {len(inv_time)}")
        print(f"  Input time length: {len(self.input.time)}")
        print(f"  Prediction time length: {len(self.prediction.time)}")
        print(f"  Truth time length: {len(self.truth.time)}")
        
        # Align all to inv time dimension and coordinates
        coords_dict = {'time': inv_time}
        if inv_lat is not None:
            coords_dict['lat'] = inv_lat
        if inv_lon is not None:
            coords_dict['lon'] = inv_lon
        
        self.input = self.input.assign_coords(coords_dict)
        self.prediction = self.prediction.assign_coords(coords_dict)
        self.truth = self.truth.assign_coords(coords_dict)
        
        # Apply DWD mask to prediction and truth (Germany region only)
        if 'dwd_mask' in self.input.data_vars:
            print("Applying DWD mask (Germany region) to prediction and truth...")
            dwd_mask = self.input['dwd_mask'].isel(time=0)  # Mask is time-invariant
            self.prediction = self.prediction.where(dwd_mask == 1)
            self.truth = self.truth.where(dwd_mask == 1)
            print("  DWD mask applied - data outside Germany set to NaN")
        
        # Add units attributes to all groups
        temp_vars = ['tas', 'tasmin', 'tasmax']
        # Input group
        for var in temp_vars:
            if var in self.input.data_vars:
                self.input[var].attrs['units'] = 'degC'
        if 'pr' in self.input.data_vars:
            self.input['pr'].attrs['units'] = 'mm/day'
        
        # Prediction group
        for var in temp_vars:
            if var in self.prediction.data_vars:
                self.prediction[var].attrs['units'] = 'degC'
        if 'pr' in self.prediction.data_vars:
            self.prediction['pr'].attrs['units'] = 'mm/day'
        
        # Truth group
        for var in temp_vars:
            if var in self.truth.data_vars:
                self.truth[var].attrs['units'] = 'degC'
        if 'pr' in self.truth.data_vars:
            self.truth['pr'].attrs['units'] = 'mm/day'
        
        self.years = sorted(found_years)
        
        print(f"1km data loaded - Time range: {self.inv.time.min().values} to {self.inv.time.max().values}")
        print(f"1km grid shape: {self.inv.lat.isel(time=0).shape}")
        
        return self
    
    def extract_GHCN(self, ds_obs, variable):
        """
        Extract variable data at GHCN station locations
        
        Parameters:
        -----------
        ds_obs : xarray.Dataset
            GHCN observations dataset (load using load_GHCN function)
        variable : str
            Variable name to extract (e.g., 'pr', 'tas', 'tasmin', 'tasmax')
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'observations': xarray.DataArray with GHCN observations for the variable
            - 'model_input': xarray.DataArray with model input at stations
            - 'model_prediction': xarray.DataArray with model predictions at stations
            - 'model_truth': xarray.DataArray with truth data at stations
            - 'indices': tuple of (y_indices, x_indices) for station locations
            - 'variable': name of the extracted variable
        """
        if self.input is None or self.prediction is None or self.inv is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        # Extract model data at stations
        print(f"\n=== Extracting {variable} from MSWXDWD at GHCN stations ===")
        ds_obs_aligned, model_input_stations, model_prediction_stations, indices = \
            extract_model_at_stations(
                self.input,
                self.prediction,
                self.inv,
                ds_obs,
                "MSWXDWD-1km"
            )
        
        # Extract truth data at stations
        truth_normalized = self.truth.assign_coords(time=pd.to_datetime(self.truth.time.values))
        obs_times = pd.to_datetime(ds_obs_aligned.time.values)
        common_times = obs_times.intersection(pd.to_datetime(self.inv.time.values))
        
        try:
            truth_aligned = truth_normalized.sel(time=common_times)
        except (KeyError, TypeError):
            truth_mask = truth_normalized.time.isin(common_times)
            truth_aligned = truth_normalized.where(truth_mask, drop=True)
        
        y_indices, x_indices = indices
        y_indices_da = xr.DataArray(y_indices, dims="station")
        x_indices_da = xr.DataArray(x_indices, dims="station")
        model_truth_stations = truth_aligned.isel(x=x_indices_da, y=y_indices_da)
        
        print(f"\n=== Extraction complete ===")
        print(f"Variable: {variable}")
        print(f"Stations: {len(ds_obs_aligned.station)}")
        print(f"Timesteps: {len(ds_obs_aligned.time)}")
        
        return {
            'observations': ds_obs_aligned[variable],
            'model_input': model_input_stations[variable],
            'model_prediction': model_prediction_stations[variable],
            'model_truth': model_truth_stations[variable],
            'indices': indices,
            'variable': variable
        }


# ==================== Helper Functions ====================

def extract_dataset_metadata(ds):
    """Extract spatial and temporal metadata from dataset"""
    start_date = pd.to_datetime(ds.time.min().values)
    end_date = pd.to_datetime(ds.time.max().values)
    
    # Handle time-varying vs time-invariant coordinates
    if 'time' in ds.lon.dims:
        XLON = ds.lon.isel(time=-1).values
        XLAT = ds.lat.isel(time=-1).values
    else:
        XLON = ds.lon.values
        XLAT = ds.lat.values
    
    lat_min, lat_max = float(XLAT.min()), float(XLAT.max())
    lon_min, lon_max = float(XLON.min()), float(XLON.max())
    
    metadata = {
        'start_date': start_date,
        'end_date': end_date,
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lon_min': lon_min,
        'lon_max': lon_max,
        'num_timesteps': len(ds.time),
    }
    
    return metadata

def filter_ghcn_stations_by_metadata(metadata, ghcn_base_path="/data01/FDS/muduchuru/Atmos/GHCN", 
                                     elements=['TMAX', 'TMIN', 'TAVG','PRCP'], verbose=True):
    """Filter GHCN stations based on spatial and temporal metadata"""
    # Extract bounds from metadata
    lat_min = metadata['lat_min']
    lat_max = metadata['lat_max']
    lon_min = metadata['lon_min']
    lon_max = metadata['lon_max']
    time_start = metadata['start_date']
    time_end = metadata['end_date']
    year_start = time_start.year
    year_end = time_end.year
    
    if verbose:
        print(f"Filtering GHCN stations:")
        print(f"  Domain: {lat_min:.2f}°N-{lat_max:.2f}°N, {lon_min:.2f}°E-{lon_max:.2f}°E")
        print(f"  Time: {year_start}-{year_end}")
    
    # Read ghcnd-stations.txt
    stations_file = os.path.join(ghcn_base_path, "ghcnd-stations.txt")
    stations_df = pd.read_fwf(
        stations_file,
        colspecs=[(0,11), (12,20), (21,30), (31,37), (38,40), (41,71)],
        names=['Station_ID', 'Latitude', 'Longitude', 'Elevation', 'State', 'Name']
    )
    
    # Filter by spatial bounds
    filtered_stations = stations_df[
        (stations_df['Latitude'] >= lat_min) &
        (stations_df['Latitude'] <= lat_max) &
        (stations_df['Longitude'] >= lon_min) &
        (stations_df['Longitude'] <= lon_max)
    ].copy()
    
    # Read inventory for temporal filtering
    inventory_file = os.path.join(ghcn_base_path, "ghcnd-inventory.txt")
    if os.path.exists(inventory_file):
        inventory_df = pd.read_fwf(
            inventory_file,
            colspecs=[(0,11), (12,20), (21,30), (31,35), (36,40), (41,45)],
            names=['Station_ID', 'Latitude', 'Longitude', 'Element', 'FirstYear', 'LastYear']
        )
        
        # Filter for elements and time range
        inventory_filtered = inventory_df[
            (inventory_df['Element'].isin(elements)) &
            (inventory_df['LastYear'] >= year_start) &
            (inventory_df['FirstYear'] <= year_end)
        ]
        
        valid_station_ids = inventory_filtered['Station_ID'].unique()
        filtered_stations = filtered_stations[filtered_stations['Station_ID'].isin(valid_station_ids)]
    
    if verbose:
        print(f"  Found {len(filtered_stations)} stations with required data")
    
    return filtered_stations

def load_GHCN(metadata=None, shapefile=None, ghcn_base_path="/data01/FDS/muduchuru/Atmos/GHCN",
               elements=['TMAX', 'TMIN', 'TAVG', 'PRCP'], max_stations=10000, verbose=True):
    """
    Load and process GHCN observational data
    
    Parameters:
    -----------
    metadata : dict, optional
        Domain metadata dictionary containing 'lat_min', 'lat_max', 'lon_min', 'lon_max',
        'start_date', 'end_date'. Use extract_dataset_metadata() to get this from a model dataset.
    shapefile : str, optional
        Path to shapefile for spatial filtering (not yet implemented)
    ghcn_base_path : str
        Base path to GHCN data directory
    elements : list
        GHCN elements to filter for (e.g., ['PRCP', 'TMAX', 'TMIN', 'TAVG'])
    max_stations : int
        Maximum number of stations to load
    verbose : bool
        Print progress information
        
    Returns:
    --------
    xarray.Dataset
        GHCN observations with variables: pr, tas, tasmin, tasmax
        
    Note:
    -----
    Either metadata or shapefile must be provided.
    """
    if metadata is None and shapefile is None:
        raise ValueError("Either metadata or shapefile must be provided")
    
    if shapefile is not None:
        raise NotImplementedError("Shapefile filtering is not yet implemented. Use metadata for now.")
    
    # Filter stations based on metadata
    ghcn_stations_df = filter_ghcn_stations_by_metadata(
        metadata,
        ghcn_base_path=ghcn_base_path,
        elements=elements,
        verbose=verbose
    )
    
    # Path to daily data
    GHCNd_path = "/data01/FDS/muduchuru/Atmos/GHCN/GHCNd/"
    
    # Limit to manageable number of stations for processing
    if len(ghcn_stations_df) > max_stations:
        if verbose:
            print(f"Limiting to {max_stations} stations for processing efficiency")
        ghcn_stations_df = ghcn_stations_df.head(max_stations)
    
    # Map GHCN elements to CF variable names
    element_to_var = {
        'PRCP': 'pr',
        'TAVG': 'tas',
        'TMIN': 'tasmin',
        'TMAX': 'tasmax'
    }
    
    # Determine which variables to include based on requested elements
    requested_vars = [element_to_var[elem] for elem in elements if elem in element_to_var]
    
    station_datasets = []
    
    if verbose:
        print(f"Processing {len(ghcn_stations_df)} GHCN stations...")
        print(f"Requested variables: {requested_vars}")
    
    for idx, station_row in ghcn_stations_df.iterrows():
        station_id = station_row['Station_ID']
        
        try:
            # Read station CSV
            df = pd.read_csv(GHCNd_path + station_id + '.csv')
            
            # Convert DATE and filter by time range
            df['DATE'] = pd.to_datetime(df['DATE'])
            df = df[
                (df['DATE'] >= metadata['start_date']) &
                (df['DATE'] <= metadata['end_date'])
            ]
            
            # Handle missing variables with NaN (even if no data for time period)
            if len(df) == 0:
                # Create empty dataframe with time range and NaN values for requested elements
                time_range = pd.date_range(metadata['start_date'], 
                                         metadata['end_date'], freq='D')
                empty_data = {'DATE': time_range}
                for elem in elements:
                    empty_data[elem] = np.nan
                df = pd.DataFrame(empty_data)
            
            # Unit conversions - only for requested elements
            df_vars = {'DATE': df['DATE']}
            if 'pr' in requested_vars and 'PRCP' in df.columns:
                df_vars['pr'] = df['PRCP'] / 10.0
            elif 'pr' in requested_vars:
                df_vars['pr'] = np.nan
                
            if 'tas' in requested_vars and 'TAVG' in df.columns:
                df_vars['tas'] = df['TAVG'] / 10.0
            elif 'tas' in requested_vars:
                df_vars['tas'] = np.nan
                
            if 'tasmin' in requested_vars and 'TMIN' in df.columns:
                df_vars['tasmin'] = df['TMIN'] / 10.0
            elif 'tasmin' in requested_vars:
                df_vars['tasmin'] = np.nan
                
            if 'tasmax' in requested_vars and 'TMAX' in df.columns:
                df_vars['tasmax'] = df['TMAX'] / 10.0
            elif 'tasmax' in requested_vars:
                df_vars['tasmax'] = np.nan
            
            # Create dataframe with only requested variables
            df = pd.DataFrame(df_vars)
            df['station'] = station_id
            
            # Convert to xarray
            ds_station = (
                df
                .set_index(['station', 'DATE'])
                .to_xarray()
                .rename({'DATE': 'time'})
            )
            
            # Add coordinates
            ds_station['lat'] = xr.DataArray([station_row['Latitude']], dims='station')
            ds_station['lon'] = xr.DataArray([station_row['Longitude']], dims='station')
            
            station_datasets.append(ds_station)
            
        except Exception as e:
            if verbose:
                print(f"Warning: Could not read station {station_id}, creating with NaN values: {e}")
            # Create station dataset with NaN values for requested variables only
            time_range = pd.date_range(metadata['start_date'], 
                                     metadata['end_date'], freq='D')
            nan_data = {
                'DATE': time_range,
                'station': station_id
            }
            for var in requested_vars:
                nan_data[var] = np.nan
            df = pd.DataFrame(nan_data)
            
            # Convert to xarray
            ds_station = (
                df
                .set_index(['station', 'DATE'])
                .to_xarray()
                .rename({'DATE': 'time'})
            )
            
            # Add coordinates
            ds_station['lat'] = xr.DataArray([station_row['Latitude']], dims='station')
            ds_station['lon'] = xr.DataArray([station_row['Longitude']], dims='station')
            
            station_datasets.append(ds_station)
        
        if (idx + 1) % 20 == 0 and verbose:
            print(f"  Processed {idx + 1}/{len(ghcn_stations_df)} stations")
    
    # Concatenate all station datasets
    if len(station_datasets) > 0:
        if verbose:
            print(f"\nMerging {len(station_datasets)} station datasets...")
        ds_obs = xr.concat(station_datasets, dim='station')
        
        if verbose:
            print(f"GHCN dataset created:")
            print(f"  Stations: {len(ds_obs.station)}")
            print(f"  Time range: {ds_obs.time.min().values} to {ds_obs.time.max().values}")
            print(f"  Variables: {list(ds_obs.data_vars)}")
        
        return ds_obs
    else:
        raise ValueError("No observational data could be loaded")


def extract_model_at_stations(model_input, model_prediction, model_inv, ds_obs, model_name):
    """Extract model data at station locations using k-d tree"""
    print(f"\nExtracting {model_name} data at station locations...")
    
    # Convert time coordinates to datetime following project standards
    obs_times = pd.to_datetime(ds_obs.time.values)
    model_times = pd.to_datetime(model_inv.time.values)
    
    # Find common times
    common_times = obs_times.intersection(model_times)
    print(f"  Common timesteps: {len(common_times)}")
    
    if len(common_times) == 0:
        raise ValueError(f"No common timesteps found for {model_name}")
    
    # Critical project pattern: Normalize time coordinates before selection
    ds_obs_normalized = ds_obs.assign_coords(time=pd.to_datetime(ds_obs.time.values))
    model_input_normalized = model_input.assign_coords(time=pd.to_datetime(model_input.time.values))
    model_prediction_normalized = model_prediction.assign_coords(time=pd.to_datetime(model_prediction.time.values))
    model_inv_normalized = model_inv.assign_coords(time=pd.to_datetime(model_inv.time.values))
    
    # Subset datasets to common times using normalized coordinates
    try:
        ds_obs_aligned = ds_obs_normalized.sel(time=common_times)
        model_input_aligned = model_input_normalized.sel(time=common_times)
        model_prediction_aligned = model_prediction_normalized.sel(time=common_times)
        model_inv_aligned = model_inv_normalized.sel(time=common_times)
    except (KeyError, TypeError) as e:
        # Fallback: Use isin for more robust time selection
        print(f"  Using fallback time selection method...")
        obs_mask = ds_obs_normalized.time.isin(common_times)
        model_mask = model_inv_normalized.time.isin(common_times)
        
        ds_obs_aligned = ds_obs_normalized.where(obs_mask, drop=True)
        model_input_aligned = model_input_normalized.where(model_mask, drop=True)
        model_prediction_aligned = model_prediction_normalized.where(model_mask, drop=True)
        model_inv_aligned = model_inv_normalized.where(model_mask, drop=True)
    
    # Critical fix: Handle different coordinate structures between models
    # Try to get coordinates from different possible locations
    print(f"  Available coordinates in model_inv: {list(model_inv_aligned.coords.keys())}")
    
    # Method 1: Try time-varying coordinates first
    try:
        if 'time' in model_inv_aligned.lat.dims:
            model_lat = model_inv_aligned.lat.isel(time=-1).values
            model_lon = model_inv_aligned.lon.isel(time=-1).values
            print("  Using time-varying coordinates")
        else:
            model_lat = model_inv_aligned.lat.values
            model_lon = model_inv_aligned.lon.values
            print("  Using time-invariant coordinates")
    except (AttributeError, KeyError):
        # Method 2: Try to get from data variables if not in coordinates
        try:
            model_lat = model_inv_aligned['lat'].isel(time=-1).values if 'time' in model_inv_aligned['lat'].dims else model_inv_aligned['lat'].values
            model_lon = model_inv_aligned['lon'].isel(time=-1).values if 'time' in model_inv_aligned['lon'].dims else model_inv_aligned['lon'].values
            print("  Using coordinates from data variables")
        except KeyError as e:
            raise ValueError(f"Cannot find lat/lon coordinates in {model_name}: {e}")
    
    print(f"  Raw coordinate shapes - lat: {model_lat.shape}, lon: {model_lon.shape}")
    
    # Handle different coordinate array structures
    if model_lat.ndim == 1 and model_lon.ndim == 1:
        # Case 1: 1D coordinate arrays - create meshgrid
        print("  Creating meshgrid from 1D coordinates")
        model_lon_2d, model_lat_2d = np.meshgrid(model_lon, model_lat)
    elif model_lat.ndim == 2 and model_lon.ndim == 2:
        # Case 2: 2D coordinate arrays (time-varying grids)
        print("  Using 2D coordinate arrays directly")
        model_lat_2d = model_lat
        model_lon_2d = model_lon
    elif model_lat.ndim == 3 or model_lon.ndim == 3:
        # Case 3: 3D arrays - take a 2D slice
        print("  Extracting 2D slice from 3D coordinate arrays")
        if model_lat.ndim == 3:
            model_lat_2d = model_lat[0, :, :] if model_lat.shape[0] < model_lat.shape[-1] else model_lat[:, :, 0]
        else:
            model_lat_2d = model_lat
        if model_lon.ndim == 3:
            model_lon_2d = model_lon[0, :, :] if model_lon.shape[0] < model_lon.shape[-1] else model_lon[:, :, 0]
        else:
            model_lon_2d = model_lon
    else:
        raise ValueError(f"Unsupported coordinate dimensions - lat: {model_lat.shape}, lon: {model_lon.shape}")
    
    print(f"  Final model grid shape: {model_lat_2d.shape}")
    
    # Station coordinates - follow original notebook pattern for proper extraction
    print(f"  Station coordinate shapes - lat: {ds_obs_aligned.lat.shape}, lon: {ds_obs_aligned.lon.shape}")
    print(f"  Station dataset dims: {ds_obs_aligned.dims}")
    
    # Handle different possible coordinate structures for stations
    if ds_obs_aligned.lat.ndim == 1 and ds_obs_aligned.lon.ndim == 1:
        # Case 1: 1D station coordinates (standard case)
        station_lats = ds_obs_aligned.lat.values
        station_lons = ds_obs_aligned.lon.values
        print("  Using 1D station coordinates")
    elif ds_obs_aligned.lat.ndim == 2 and ds_obs_aligned.lon.ndim == 2:
        # Case 2: 2D station coordinates - take diagonal or first row/column
        station_lats = np.diag(ds_obs_aligned.lat.values) if ds_obs_aligned.lat.shape[0] == ds_obs_aligned.lat.shape[1] else ds_obs_aligned.lat.values[:, 0]
        station_lons = np.diag(ds_obs_aligned.lon.values) if ds_obs_aligned.lon.shape[0] == ds_obs_aligned.lon.shape[1] else ds_obs_aligned.lon.values[:, 0]
        print("  Using 2D station coordinates - extracted diagonal/first column")
    else:
        # Fallback: Try to get coordinate values by station dimension
        try:
            if 'station' in ds_obs_aligned.lat.dims:
                station_lats = ds_obs_aligned.lat.values
                station_lons = ds_obs_aligned.lon.values
                print("  Using station-indexed coordinates")
            else:
                # Last resort: flatten and take unique values (assuming repeated coordinates)
                station_lats_flat = ds_obs_aligned.lat.values.ravel()
                station_lons_flat = ds_obs_aligned.lon.values.ravel()
                
                # Get unique coordinate pairs
                unique_coords = np.unique(np.column_stack([station_lats_flat, station_lons_flat]), axis=0)
                station_lats = unique_coords[:, 0]
                station_lons = unique_coords[:, 1]
                print(f"  Using flattened unique coordinates: {len(unique_coords)} unique pairs")
        except Exception as e:
            raise ValueError(f"Cannot extract station coordinates properly: {e}")
    
    print(f"  Extracted station coordinates - lat shape: {station_lats.shape}, lon shape: {station_lons.shape}")
    print(f"  Number of stations: {len(station_lats)}")
    
    # Ensure station coordinates are 1D
    if station_lats.ndim > 1:
        station_lats = station_lats.ravel()
    if station_lons.ndim > 1:
        station_lons = station_lons.ravel()
    
    # Build k-d tree for nearest neighbor search (critical project pattern)
    # Ensure proper flattening and point formation
    model_points = np.column_stack([
        model_lat_2d.ravel(),
        model_lon_2d.ravel()
    ])
    station_points = np.column_stack([station_lats, station_lons])
    
    print(f"  Model points shape: {model_points.shape}")
    print(f"  Station points shape: {station_points.shape}")
    
    # Validate point arrays before k-d tree
    if model_points.shape[1] != 2:
        raise ValueError(f"Model points must have 2 columns (lat, lon), got shape {model_points.shape}")
    if station_points.shape[1] != 2:
        raise ValueError(f"Station points must have 2 columns (lat, lon), got shape {station_points.shape}")
    if len(station_lats) != len(station_lons):
        raise ValueError(f"Mismatch in station coordinate lengths: lat={len(station_lats)}, lon={len(station_lons)}")
    
    kdtree = cKDTree(model_points)
    distances, nearest_indices = kdtree.query(station_points)
    
    # Convert to 2D indices
    y_indices, x_indices = np.unravel_index(nearest_indices, model_lat_2d.shape)
    
    print(f"  Mean distance to nearest grid point: {distances.mean():.3f} degrees")
    print(f"  Max distance to nearest grid point: {distances.max():.3f} degrees")
    
    # Extract model data at station locations using standard indexing pattern
    y_indices_da = xr.DataArray(y_indices, dims="station")
    x_indices_da = xr.DataArray(x_indices, dims="station")
    
    # Detect dimension names (x/y or lat/lon)
    if 'x' in model_input_aligned.dims and 'y' in model_input_aligned.dims:
        model_input_stations = model_input_aligned.isel(x=x_indices_da, y=y_indices_da)
        model_prediction_stations = model_prediction_aligned.isel(x=x_indices_da, y=y_indices_da)
    elif 'lat' in model_input_aligned.dims and 'lon' in model_input_aligned.dims:
        model_input_stations = model_input_aligned.isel(lat=y_indices_da, lon=x_indices_da)
        model_prediction_stations = model_prediction_aligned.isel(lat=y_indices_da, lon=x_indices_da)
    else:
        raise ValueError(f"Cannot detect spatial dimensions in {model_name}. Expected 'x'/'y' or 'lat'/'lon'")
    
    return ds_obs_aligned, model_input_stations, model_prediction_stations, (y_indices, x_indices)

def assign_season(time_coord):
    """Assign season based on month"""
    month = pd.to_datetime(time_coord).month
    conditions = [
        (month.isin([12, 1, 2])),    # DJF
        (month.isin([3, 4, 5])),     # MAM  
        (month.isin([6, 7, 8])),     # JJA
        (month.isin([9, 10, 11]))    # SON
    ]
    choices = ['DJF', 'MAM', 'JJA', 'SON']
    return np.select(conditions, choices)


# ==================== Optimized Extraction Functions ====================

def build_kdtree(model_inv):
    """Build KDTree once from model grid (lazy-safe).
    
    This is a critical optimization - building the KDTree once and reusing it
    provides 2-3x speedup compared to rebuilding for each extraction.
    
    Parameters:
    -----------
    model_inv : xarray.Dataset
        Model inventory dataset containing lat/lon coordinates
        
    Returns:
    --------
    kdtree : scipy.spatial.cKDTree
        Spatial index for fast nearest neighbor queries
    grid_shape : tuple
        Shape of the 2D grid (for unraveling indices)
    """
    print("Building KDTree...")

    # Handle time-varying vs time-invariant coordinates
    if 'time' in model_inv.lat.dims:
        lat = model_inv.lat.isel(time=0).compute().values
        lon = model_inv.lon.isel(time=0).compute().values
    else:
        lat = model_inv.lat.compute().values
        lon = model_inv.lon.compute().values

    # Handle 1D vs 2D
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lat2d, lon2d = lat, lon

    points = np.column_stack([lat2d.ravel(), lon2d.ravel()])
    kdtree = cKDTree(points)

    print(f"KDTree built with {points.shape[0]} grid points")

    return kdtree, lat2d.shape


def query_station_indices(kdtree, grid_shape, ds_obs):
    """Find nearest grid point indices for all stations.
    
    Vectorized operation that queries all stations at once - much faster
    than querying one at a time.
    
    Parameters:
    -----------
    kdtree : scipy.spatial.cKDTree
        Pre-built spatial index from build_kdtree()
    grid_shape : tuple
        Shape of the 2D grid
    ds_obs : xarray.Dataset
        Observations dataset with lat/lon coordinates
        
    Returns:
    --------
    y_idx : xarray.DataArray
        Y indices for all stations (dims: station)
    x_idx : xarray.DataArray  
        X indices for all stations (dims: station)
    """
    print("Querying station locations...")

    station_lats = ds_obs.lat.values
    station_lons = ds_obs.lon.values

    station_points = np.column_stack([station_lats, station_lons])

    distances, indices = kdtree.query(station_points)

    y_idx, x_idx = np.unravel_index(indices, grid_shape)

    print(f"Mean distance: {distances.mean():.3f}")

    return xr.DataArray(y_idx, dims="station"), xr.DataArray(x_idx, dims="station")


def extract_at_indices(data, y_idx, x_idx):
    """Fast vectorized extraction at pre-computed indices.
    
    Avoids using .values which forces computation. Stays lazy until
    explicitly computed or persisted.
    
    Parameters:
    -----------
    data : xarray.DataArray
        Data array to extract from
    y_idx : xarray.DataArray
        Y indices (dims: station)
    x_idx : xarray.DataArray
        X indices (dims: station)
        
    Returns:
    --------
    xarray.DataArray
        Extracted data at station locations (dims: time, station, ...)
    """
    # Detect dimension names (x/y or lat/lon)
    if 'x' in data.dims and 'y' in data.dims:
        return data.isel(y=y_idx, x=x_idx)
    elif 'lat' in data.dims and 'lon' in data.dims:
        return data.isel(lat=y_idx, lon=x_idx)
    else:
        raise ValueError(f"Cannot detect spatial dimensions. Expected 'x'/'y' or 'lat'/'lon', got {data.dims}")


def prepare_data(data):
    """Rechunk and persist data for optimal station extraction.
    
    Chunking strategy:
    - time: -1 (full time dimension in one chunk for fast time-series access)
    - station: 100 (batched for parallel processing)
    
    Parameters:
    -----------
    data : xarray.DataArray
        Data to prepare
        
    Returns:
    --------
    xarray.DataArray
        Rechunked and persisted data in cluster memory
    """
    return data.chunk({'time': -1, 'station': 100}).persist()
