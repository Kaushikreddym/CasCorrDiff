"""
Visualization utilities for climate model validation.

Contains helper functions for creating various validation plots:
- Seasonal maps
- Metric maps
- Quantile difference plots
- 2D density scatter plots with KDE
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
from matplotlib.colors import PowerNorm
from scipy.stats import pearsonr, spearmanr, kendalltau, wasserstein_distance
import colormaps as cmaps


def calculate_metrics(obs_data, pred_data, wet_threshold=1.0):
    """
    Calculate multiple metrics for paired time series data at each station:
    - RMSE, Spearman ρ, Kendall τ, Wasserstein Distance
    - Frequency Bias, POD, FAR, CSI
    - Extreme Bias (90th percentile difference)
    
    Parameters:
    -----------
    obs_data : ndarray
        Observed data with shape (n_stations, n_timesteps)
    pred_data : ndarray
        Predicted data with shape (n_stations, n_timesteps)
    wet_threshold : float
        Threshold for wet day detection (default: 1.0 mm/day for precip)
    
    Returns:
    --------
    metrics : dict
        Dictionary containing arrays of metrics for each station
    """
    n_stations = obs_data.shape[0]
    
    metrics = {
        'rmse': np.full(n_stations, np.nan),
        'spearman': np.full(n_stations, np.nan),
        'kendall': np.full(n_stations, np.nan),
        'wasserstein': np.full(n_stations, np.nan),
        'freq_bias': np.full(n_stations, np.nan),
        'pod': np.full(n_stations, np.nan),
        'far': np.full(n_stations, np.nan),
        'csi': np.full(n_stations, np.nan),
        'extreme_bias': np.full(n_stations, np.nan)
    }
    
    for i in range(n_stations):
        obs_station = obs_data[i, :]
        pred_station = pred_data[i, :]
        
        # Find valid pairs (both obs and pred non-NaN)
        valid_mask = ~np.isnan(obs_station) & ~np.isnan(pred_station)
        
        if valid_mask.sum() < 10:  # Need at least 10 valid pairs
            continue
            
        obs_valid = obs_station[valid_mask]
        pred_valid = pred_station[valid_mask]
        
        # RMSE
        metrics['rmse'][i] = np.sqrt(np.mean((pred_valid - obs_valid)**2))
        
        # Spearman correlation
        if len(obs_valid) > 1:
            rho, _ = spearmanr(obs_valid, pred_valid)
            metrics['spearman'][i] = rho
        
        # Kendall tau
        if len(obs_valid) > 1:
            tau, _ = kendalltau(obs_valid, pred_valid)
            metrics['kendall'][i] = tau
        
        # Wasserstein distance
        metrics['wasserstein'][i] = wasserstein_distance(obs_valid, pred_valid)
        
        # Wet day detection metrics
        obs_wet = obs_valid > wet_threshold
        pred_wet = pred_valid > wet_threshold
        
        # Contingency table
        hits = np.sum(obs_wet & pred_wet)  # True positives
        misses = np.sum(obs_wet & ~pred_wet)  # False negatives
        false_alarms = np.sum(~obs_wet & pred_wet)  # False positives
        
        # Frequency Bias
        obs_wet_count = np.sum(obs_wet)
        pred_wet_count = np.sum(pred_wet)
        if obs_wet_count > 0:
            metrics['freq_bias'][i] = pred_wet_count / obs_wet_count
        
        # Probability of Detection (POD)
        if (hits + misses) > 0:
            metrics['pod'][i] = hits / (hits + misses)
        
        # False Alarm Ratio (FAR)
        if (hits + false_alarms) > 0:
            metrics['far'][i] = false_alarms / (hits + false_alarms)
        
        # Critical Success Index (CSI)
        if (hits + misses + false_alarms) > 0:
            metrics['csi'][i] = hits / (hits + misses + false_alarms)
        
        # Extreme Bias (90th percentile difference)
        if len(obs_valid) > 80:
            p90_obs = np.percentile(obs_valid, 90)
            p90_pred = np.percentile(pred_valid, 90)
            metrics['extreme_bias'][i] = (p90_pred - p90_obs)
    
    return metrics


def calculate_metrics_and_means(obs_data, model_data, threshold=0.1):
    """
    Calculate performance metrics on raw time series data, then return time means for plotting.
    
    Parameters:
    -----------
    obs_data : xarray.DataArray
        Observed data
    model_data : xarray.DataArray
        Model data
    threshold : float
        Minimum threshold for valid data (default: 0.1)
    
    Returns:
    --------
    obs_valid : ndarray
        Valid observed station means
    model_valid : ndarray
        Valid model station means
    metrics : dict
        Dictionary of performance metrics
    """
    # Flatten and remove NaN values for metric calculation
    obs_flat = obs_data.values.flatten()
    model_flat = model_data.values.flatten()
    
    # Create mask for valid pairs
    valid_mask = ~np.isnan(obs_flat) & ~np.isnan(model_flat) & (obs_flat > threshold)
    
    obs_valid_all = obs_flat[valid_mask]
    model_valid_all = model_flat[valid_mask]
    
    if len(obs_valid_all) < 100:
        return None, None, None
    
    # Calculate metrics on RAW data
    r, _ = pearsonr(obs_valid_all, model_valid_all)
    
    # R² (square of Pearson correlation — variance explained by linear relationship)
    r_squared = r ** 2
    
    # Nash-Sutcliffe Efficiency (uses obs mean as baseline, penalises bias too)
    mean_obs = np.mean(obs_valid_all)
    SS_res = np.sum((obs_valid_all - model_valid_all)**2)
    SS_tot = np.sum((obs_valid_all - mean_obs)**2)
    nse = 1 - (SS_res / SS_tot)
    
    # Percent Error
    percent_error = 100 * (np.mean(model_valid_all) - np.mean(obs_valid_all)) / np.mean(obs_valid_all)
    
    metrics = {
        'r': r,
        'r2': r_squared,
        'nse': nse,
        'percent_error': percent_error
    }
    
    # Take mean across time for plotting
    obs_mean = np.nanmean(obs_data.values, axis=1)
    model_mean = np.nanmean(model_data.values, axis=1)
    
    valid_mask_stations = ~np.isnan(obs_mean) & ~np.isnan(model_mean) & (obs_mean > threshold)
    
    obs_valid = obs_mean[valid_mask_stations]
    model_valid = model_mean[valid_mask_stations]
    
    return obs_valid, model_valid, metrics


def calculate_quantile_diff(obs_data, model_data, n_quantiles=60, threshold=0.1):
    """
    Calculate quantile difference data (model - obs) at each quantile level.

    Strategy: for each station compute quantile differences over its time series,
    then average those per-station differences across all valid stations.
    This avoids mixing spatial and temporal variability.

    Parameters:
    -----------
    obs_data : xarray.DataArray
        Observed data, shape (station, time)
    model_data : xarray.DataArray
        Model data, shape (station, time)
    n_quantiles : int
        Number of quantile levels to calculate
    threshold : float
        Minimum threshold for valid data (wet-day filter)

    Returns:
    --------
    quantile_levels : ndarray
        Quantile levels (percentiles)
    quantile_diff : ndarray
        Mean difference (model - obs) averaged across stations at each quantile
    obs_quantiles : ndarray
        Mean observed quantile values averaged across stations
    """
    quantile_levels = np.linspace(5, 95, n_quantiles)

    obs_arr   = obs_data.values    # (station, time)
    model_arr = model_data.values  # (station, time)

    n_stations = obs_arr.shape[0]
    station_obs_q   = np.full((n_stations, n_quantiles), np.nan)
    station_diff_q  = np.full((n_stations, n_quantiles), np.nan)

    for i in range(n_stations):
        obs_t   = obs_arr[i, :]
        model_t = model_arr[i, :]

        # Valid pairs where obs exceeds wet-day threshold
        valid = ~np.isnan(obs_t) & ~np.isnan(model_t) & (obs_t > threshold)
        if valid.sum() < 20:
            continue

        obs_v   = obs_t[valid]
        model_v = model_t[valid]

        obs_q   = np.percentile(obs_v,   quantile_levels)
        model_q = np.percentile(model_v, quantile_levels)

        station_obs_q[i, :]  = obs_q
        station_diff_q[i, :] = model_q - obs_q

    # Average across stations (ignore stations that had no valid data)
    obs_quantiles  = np.nanmean(station_obs_q,  axis=0)
    quantile_diff  = np.nanmean(station_diff_q, axis=0)

    n_valid = np.sum(~np.isnan(station_obs_q[:, 0]))
    if n_valid < 5:
        return None, None, None

    return quantile_levels, quantile_diff, obs_quantiles


def plot_seasonal_maps(datasets, col_titles, ds_obs, output_path=None, 
                       variable_name='pr', variable_units='mm/day'):
    """
    Create 4x6 panel seasonal maps.
    
    Parameters:
    -----------
    datasets : list
        List of xarray DataArrays with seasonal data (must have 'group' dimension)
    col_titles : list
        List of column titles
    ds_obs : xarray.Dataset
        Dataset containing station lat/lon coordinates
    output_path : str, optional
        Path to save figure
    variable_name : str
        Variable name for colorbar label
    variable_units : str
        Units for colorbar label
    """
    station_lats = ds_obs.lat.values
    station_lons = ds_obs.lon.values
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    
    proj = ccrs.LambertConformal(central_longitude=10.5, central_latitude=51.0)
    fig, axes = plt.subplots(4, len(datasets), figsize=(6*len(datasets), 20),
                             subplot_kw={'projection': proj})
    
    # Get colormap range
    all_data = np.concatenate([d.values.flatten() for d in datasets])
    vmin, vmax = np.nanpercentile(all_data, [2, 98])
    norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    
    for row, season in enumerate(seasons):
        for col, (data, title) in enumerate(zip(datasets, col_titles)):
            ax = axes[row, col]
            
            ax.add_feature(cfeature.BORDERS, linewidth=1.5, edgecolor='black')
            ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black')
            
            season_data = data.sel(group=season)
            valid_mask_season = ~np.isnan(season_data.values)
            
            scatter = ax.scatter(
                station_lons[valid_mask_season],
                station_lats[valid_mask_season],
                c=season_data.values[valid_mask_season],
                s=100,
                cmap=cmaps.precip2_17lev if variable_name == 'pr' else 'RdYlBu_r',
                norm=norm,
                edgecolors=None,
                linewidths=0.,
                transform=ccrs.PlateCarree()
            )
            
            lat_buffer = (station_lats.max() - station_lats.min()) * 0.1
            lon_buffer = (station_lons.max() - station_lons.min()) * 0.1
            ax.set_extent([
                station_lons.min() - lon_buffer,
                station_lons.max() + lon_buffer,
                station_lats.min() - lat_buffer,
                station_lats.max() + lat_buffer
            ])
    
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    if col != 0:
        gl.left_labels = False
    else:
        gl.ylabel_style = {'size': 28}
    
    if row != 3:
        gl.bottom_labels = False
    else:
        gl.xlabel_style = {'size': 28}
    
    gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
    gl.xlocator = plt.MultipleLocator(5)
    gl.ylocator = plt.MultipleLocator(5)
    
    if row == 0:
        # Create letter label (a, b, c, d, ...)
        letter_label = chr(97 + col)  # 97 is 'a'
        ax.set_title(f'{letter_label}) {title}', fontsize=28, fontweight='bold', pad=10)
    
    if col == 0:
        ax.text(-0.35, 0.5, season, transform=ax.transAxes,
               fontsize=28, fontweight='bold', va='center', rotation=90)
    
    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes([1.05, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label(f'{variable_name} ({variable_units})', fontsize=28, fontweight='bold')
    cbar.ax.tick_params(labelsize=26)
    
    if output_path:
        # Convert PosixPath to string and save in both PNG and SVG formats
        output_path = str(output_path)
        base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
        plt.savefig(f'{base_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{base_path}.svg', dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_metric_maps(all_metrics, model_names, metric_info, ds_obs, output_path=None):
    """
    Create metric maps for all models.
    
    Parameters:
    -----------
    all_metrics : list
        List of metric dictionaries from calculate_metrics
    model_names : list
        List of model names
    metric_info : dict
        Dictionary with 'name', 'key', 'cmap', 'center', 'vmin', 'vmax'
    ds_obs : xarray.Dataset
        Dataset containing station coordinates
    output_path : str, optional
        Path to save figure
    """
    station_lats = ds_obs.lat.values
    station_lons = ds_obs.lon.values
    
    metric_name = metric_info['name']
    metric_key = metric_info['key']
    
    proj = ccrs.LambertConformal(central_longitude=10.5, central_latitude=51.0)
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 8),
                             subplot_kw={'projection': proj})
    
    # Collect all values for colorbar range
    all_values = []
    for metrics in all_metrics:
        all_values.extend(metrics[metric_key][~np.isnan(metrics[metric_key])])
    
    if 'vmin' in metric_info and 'vmax' in metric_info:
        vmin, vmax = metric_info['vmin'], metric_info['vmax']
    elif metric_info.get('center') is not None:
        max_abs = max(abs(np.nanpercentile(all_values, 5) - metric_info['center']),
                     abs(np.nanpercentile(all_values, 95) - metric_info['center']))
        vmin = metric_info['center'] - max_abs
        vmax = metric_info['center'] + max_abs
    else:
        vmin, vmax = np.nanpercentile(all_values, [5, 95])
    
    for col, (model_name, metrics) in enumerate(zip(model_names, all_metrics)):
        ax = axes.flatten()[col]
        
        ax.add_feature(cfeature.BORDERS, linewidth=1.0, edgecolor='black')
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black')
        
        metric_values = metrics[metric_key]
        valid_mask = ~np.isnan(metric_values)
        
        if valid_mask.sum() == 0:
            ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=16)
            continue
        
        scatter = ax.scatter(
            station_lons[valid_mask],
            station_lats[valid_mask],
            c=metric_values[valid_mask],
            s=350,
            cmap=metric_info['cmap'],
            vmin=vmin,
            vmax=vmax,
            edgecolors='black',
            linewidths=0.,
            transform=ccrs.PlateCarree()
        )
        
        lat_buffer = (station_lats.max() - station_lats.min()) * 0.1
        lon_buffer = (station_lons.max() - station_lons.min()) * 0.1
        ax.set_extent([
            station_lons.min() - lon_buffer,
            station_lons.max() + lon_buffer,
            station_lats.min() - lat_buffer,
            station_lats.max() + lat_buffer
        ])
        
        gl = ax.gridlines(draw_labels=False, alpha=0.3)
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = False  # Only show left labels on first column
        gl.bottom_labels = False  # Always show bottom labels in single row
        gl.xlabel_style = {'size': 26}
        gl.ylabel_style = {'size': 26}
        
        gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
        gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
        gl.xlocator = plt.MultipleLocator(5)
        gl.ylocator = plt.MultipleLocator(5)
        
        # Create letter label (a, b, c, d, ...)
        letter_label = chr(97 + col)  # 97 is 'a'
        ax.set_title(f'{letter_label}) {model_name}', fontsize=26, fontweight='bold', pad=10)
        ax.tick_params(axis='both', which='major', labelsize=26)
        
        # Make spines thicker and visible
        for spine in ax.spines.values():
            spine.set_linewidth(4)
            spine.set_visible(True)
            spine.set_color('black')
        
        mean_val = np.nanmean(metric_values)
        median_val = np.nanmedian(metric_values)
        ax.text(0.02, 0.98, f'Mean: {mean_val:.3f}\nMedian: {median_val:.3f}',
               transform=ax.transAxes, fontsize=26, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, linewidth=3, edgecolor='black'))
    
    fig.subplots_adjust(right=0.98)
    cax = fig.add_axes([1.02, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label(metric_name, fontsize=28, fontweight='bold')
    cbar.ax.tick_params(labelsize=26)
    
    fig.suptitle(f'{metric_name}', fontsize=32, fontweight='bold', y=1.02)
    
    if output_path:
        # Convert PosixPath to string and save in both PNG and SVG formats
        output_path = str(output_path)
        base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
        plt.savefig(f'{base_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{base_path}.svg', dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_quantile_difference(obs, model_data_list, output_path=None, 
                            variable_name='pr', variable_units='mm/day'):
    """
    Create quantile difference plot comparing multiple models.
    
    Parameters:
    -----------
    obs : xarray.DataArray
        Observed data
    model_data_list : list of tuples
        List of (name, data, marker, color) tuples
    output_path : str, optional
        Path to save figure
    variable_name : str
        Variable name for labels
    variable_units : str
        Units for labels
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    obs_quantile_values = None
    
    for model_name, model_data, marker, color in model_data_list:
        q_levels, q_diff, obs_q = calculate_quantile_diff(obs, model_data)
        
        if q_levels is None:
            print(f"Skipping {model_name} - insufficient data")
            continue
        
        if obs_quantile_values is None:
            obs_quantile_values = obs_q
            quantile_levels = q_levels
        
        ax.plot(q_levels, q_diff, linestyle='-', linewidth=2, marker=marker,
               color=color, label=model_name, alpha=0.8, markersize=6, markevery=3)
    
    ax.axhline(y=0, color='k', linestyle='--', linewidth=2, 
               label='Perfect agreement', zorder=0)
    ax.grid(True, alpha=0.3, zorder=0)
    
    ax.set_xlabel(f'Quantile (%) with station-mean observed values [{variable_units}]', 
                  fontsize=26, fontweight='bold', labelpad=50)
    ax.set_ylabel(f'Model - Observed ({variable_units})', 
                  fontsize=26, fontweight='bold')
    ax.set_title(f'Quantile Difference: Models vs GHCNd Observations ({variable_name})\n'
                 f'(per-station quantiles over time, averaged across stations)', 
                 fontsize=28, fontweight='bold', pad=15)
    
    ax.set_xlim(0, 100)
    xtick_positions = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ax.set_xticks(xtick_positions)
    
    obs_at_ticks = np.percentile(obs.values[~np.isnan(obs.values) & (obs.values > 0.1)], 
                                  xtick_positions)
    
    ax.set_xticklabels([f'{int(p)}' for p in xtick_positions])
    
    for pos, val in zip(xtick_positions, obs_at_ticks):
        ax.text(pos, -0.08, f'[{val:.1f}]', 
                ha='center', va='top', fontsize=16, 
                transform=ax.get_xaxis_transform(), 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='none', alpha=0.7))
    
    ax.legend(loc='lower left', fontsize=24, framealpha=0.9)
    ax.tick_params(axis='both', which='major', labelsize=26)
    
    # Make spines thicker and visible
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_visible(True)
        spine.set_color('black')
    
    plt.tight_layout()
    if output_path:
        # Convert PosixPath to string and save in both PNG and SVG formats
        output_path = str(output_path)
        base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
        plt.savefig(f'{base_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{base_path}.svg', dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_2d_density_scatter(obs, model_data_list, output_path=None,
                           variable_name='pr', variable_units='mm/day'):
    """
    Create 2D density scatter plots with KDE for multiple models (station means).
    
    Parameters:
    -----------
    obs : xarray.DataArray
        Observed data
    model_data_list : list of tuples
        List of (name, data, color) tuples
    output_path : str, optional
        Path to save figure
    variable_name : str
        Variable name for labels
    variable_units : str
        Units for labels
    """
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14), sharex=True, sharey=True)
    
    obs_station_means = np.nanmean(obs.values, axis=1)
    obs_valid_means = obs_station_means[~np.isnan(obs_station_means) & (obs_station_means > 0.1)]
    global_max = np.percentile(obs_valid_means, 99.5)
    
    for idx, (model_name, model_data, color) in enumerate(model_data_list):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Calculate station means for plotting
        obs_station_means = np.nanmean(obs.values, axis=1)
        model_station_means = np.nanmean(model_data.values, axis=1)
        
        valid_mask = ~np.isnan(obs_station_means) & ~np.isnan(model_station_means) & (obs_station_means > 0.1)
        obs_valid = obs_station_means[valid_mask]
        model_valid = model_station_means[valid_mask]
        
        if len(obs_valid) < 10:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=18)
            ax.set_title(model_name, fontsize=26, fontweight='bold')
            continue
        
        # Calculate metrics for each station over time, then average
        r_values = []
        r2_values = []
        nse_values = []
        percent_error_values = []
        
        valid_indices = np.where(valid_mask)[0]
        for i in valid_indices:
            obs_time_series = obs.values[i, :]
            model_time_series = model_data.values[i, :]
            
            valid_time = ~np.isnan(obs_time_series) & ~np.isnan(model_time_series) & (obs_time_series > 0.1)
            
            if valid_time.sum() < 10:
                continue
            
            obs_t = obs_time_series[valid_time]
            model_t = model_time_series[valid_time]
            
            # Calculate metrics for this station
            try:
                r_val, _ = pearsonr(obs_t, model_t)
                r_values.append(r_val)
                
                # R² = square of Pearson r (variance explained by linear fit)
                r2 = r_val ** 2
                r2_values.append(r2)
                
                # NSE uses obs mean as baseline (penalises both bias and scatter)
                mean_obs_t = np.mean(obs_t)
                SS_res = np.sum((obs_t - model_t)**2)
                SS_tot = np.sum((obs_t - mean_obs_t)**2)
                nse_val = 1 - (SS_res / SS_tot)
                nse_values.append(nse_val)
                
                pct_err = 100 * (np.mean(model_t) - np.mean(obs_t)) / np.mean(obs_t)
                percent_error_values.append(pct_err)
            except:
                continue
        
        # Average the metrics across stations
        r = np.mean(r_values) if r_values else np.nan
        r_squared = np.mean(r2_values) if r2_values else np.nan
        nse = np.mean(nse_values) if nse_values else np.nan
        percent_error = np.mean(percent_error_values) if percent_error_values else np.nan
        
        sns.scatterplot(x=obs_valid, y=model_valid, s=30, color="0", 
                       alpha=0.5, ax=ax, legend=False)
        
        sns.histplot(x=obs_valid, y=model_valid, bins=30, pthresh=0.05, 
                    cmap="Reds", ax=ax, cbar=False, alpha=0.8)
        
        ax.plot([0, global_max], [0, global_max], 'k--', linewidth=2.5, 
                label='1:1 line', zorder=10, alpha=0.9)
        
        ax.set_xlim(0, global_max)
        ax.set_ylim(0, global_max)
        
        ax.set_xlabel(f'GHCNd Observed ({variable_units})', 
                     fontsize=26, fontweight='bold')
        ax.set_ylabel(f'Model ({variable_units})', 
                     fontsize=26, fontweight='bold')
        # Create letter label (a, b, c, d, ...)
        letter_label = chr(97 + idx)  # 97 is 'a'
        ax.set_title(f'{letter_label}) {model_name}', fontsize=26, fontweight='bold', pad=10)
        
        ax.grid(True, alpha=0.3, zorder=0)
        
        metrics_text = f"r = {r:.3f}\n"
        # metrics_text += f"R² = {r_squared:.3f}\n"
        metrics_text += f"NSE = {nse:.3f}\n"
        metrics_text += f"Error = {percent_error:.1f}%"
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               fontsize=24, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', 
                        alpha=0.9, linewidth=3))
        
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both', which='major', labelsize=26)
        
        # Make spines thicker and visible
        for spine in ax.spines.values():
            spine.set_linewidth(4)
            spine.set_visible(True)
            spine.set_color('black')
    
    fig.suptitle(f'2D Density: Station Means {variable_name} - Model vs GHCNd Observations', 
                fontsize=28, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    if output_path:
        # Convert PosixPath to string and save in both PNG and SVG formats
        output_path = str(output_path)
        base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
        plt.savefig(f'{base_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{base_path}.svg', dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    
    sns.reset_defaults()


def plot_2d_density_time_means(obs, model_data_list, output_path=None,
                               variable_name='pr', variable_units='mm/day'):
    """
    Create 2D density scatter plots with KDE for multiple models (time means across stations).
    
    Parameters:
    -----------
    obs : xarray.DataArray
        Observed data with shape (n_stations, n_timesteps)
    model_data_list : list of tuples
        List of (name, data, color) tuples
    output_path : str, optional
        Path to save figure
    variable_name : str
        Variable name for labels
    variable_units : str
        Units for labels
    """
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14), sharex=True, sharey=True)
    
    obs_time_means = np.nanmean(obs.values, axis=0)
    obs_valid_means = obs_time_means[~np.isnan(obs_time_means) & (obs_time_means > 0.1)]
    global_max = np.percentile(obs_valid_means, 99.5)
    
    for idx, (model_name, model_data, color) in enumerate(model_data_list):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Calculate time means for plotting
        obs_time_means = np.nanmean(obs.values, axis=0)
        model_time_means = np.nanmean(model_data.values, axis=0)
        
        valid_mask = ~np.isnan(obs_time_means) & ~np.isnan(model_time_means) & (obs_time_means > 0.1)
        obs_valid = obs_time_means[valid_mask]
        model_valid = model_time_means[valid_mask]
        
        if len(obs_valid) < 10:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=18)
            ax.set_title(model_name, fontsize=26, fontweight='bold')
            continue
        
        # Calculate metrics for each time point over stations, then average
        r_values = []
        r2_values = []
        nse_values = []
        percent_error_values = []
        
        for j in np.where(valid_mask)[0]:
            obs_station_series = obs.values[:, j]
            model_station_series = model_data.values[:, j]
            
            valid_stations = ~np.isnan(obs_station_series) & ~np.isnan(model_station_series) & (obs_station_series > 0.1)
            
            if valid_stations.sum() < 10:
                continue
            
            obs_s = obs_station_series[valid_stations]
            model_s = model_station_series[valid_stations]
            
            # Calculate metrics for this time point
            try:
                r_val, _ = pearsonr(obs_s, model_s)
                r_values.append(r_val)
                
                SS_res = np.sum((obs_s - model_s)**2)
                SS_tot = np.sum((obs_s - np.mean(obs_s))**2)
                r2 = 1 - (SS_res / SS_tot)
                r2_values.append(r2)
                
                mean_obs_s = np.mean(obs_s)
                nse_val = 1 - (np.sum((obs_s - model_s)**2) / 
                               np.sum((obs_s - mean_obs_s)**2))
                nse_values.append(nse_val)
                
                pct_err = 100 * (np.mean(model_s) - np.mean(obs_s)) / np.mean(obs_s)
                percent_error_values.append(pct_err)
            except:
                continue
        
        # Average the metrics across time points
        r = np.mean(r_values) if r_values else np.nan
        r_squared = np.mean(r2_values) if r2_values else np.nan
        nse = np.mean(nse_values) if nse_values else np.nan
        percent_error = np.mean(percent_error_values) if percent_error_values else np.nan
        
        sns.scatterplot(x=obs_valid, y=model_valid, s=30, color="0", 
                       alpha=0.5, ax=ax, legend=False)
        
        sns.histplot(x=obs_valid, y=model_valid, bins=30, pthresh=0.05, 
                    cmap="Reds", ax=ax, cbar=False, alpha=0.8)
        
        ax.plot([0, global_max], [0, global_max], 'k--', linewidth=2.5, 
                label='1:1 line', zorder=10, alpha=0.9)
        
        ax.set_xlim(0, global_max)
        ax.set_ylim(0, global_max)
        
        ax.set_xlabel(f'GHCNd Observed ({variable_units})', 
                     fontsize=26, fontweight='bold')
        ax.set_ylabel(f'Model ({variable_units})', 
                     fontsize=26, fontweight='bold')
        # Create letter label (a, b, c, d, ...)
        letter_label = chr(97 + idx)  # 97 is 'a'
        ax.set_title(f'{letter_label}) {model_name}', fontsize=26, fontweight='bold', pad=10)
        
        ax.grid(True, alpha=0.3, zorder=0)
        
        metrics_text = f"r = {r:.3f}\n"
        metrics_text += f"R² = {r_squared:.3f}\n"
        metrics_text += f"NSE = {nse:.3f}\n"
        metrics_text += f"Error = {percent_error:.1f}%"
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               fontsize=24, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', 
                        alpha=0.9, linewidth=3))
        
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both', which='major', labelsize=26)
        
        # Make spines thicker and visible
        for spine in ax.spines.values():
            spine.set_linewidth(4)
            spine.set_visible(True)
            spine.set_color('black')
    
    fig.suptitle(f'2D Density: Time Means (across stations) {variable_name} - Model vs GHCNd Observations', 
                fontsize=28, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    if output_path:
        # Convert PosixPath to string and save in both PNG and SVG formats
        output_path = str(output_path)
        base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
        plt.savefig(f'{base_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{base_path}.svg', dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    
    sns.reset_defaults()

# All figures saved in both PNG (raster, 300 DPI) and SVG (vector) formats for maximum compatibility and scalability
