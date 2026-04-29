"""
CasCorrDiff Datasets Package

This package provides classes for loading and processing weather model data
and extracting it at GHCN observation station locations.
"""

from .data import (
    ERA5MSWX,
    MSWXDWD,
    ISIMIP_ERA5,
    load_GHCN,
    extract_dataset_metadata,
    filter_ghcn_stations_by_metadata,
    extract_model_at_stations,
    assign_season,
    # Optimized extraction functions
    build_kdtree,
    query_station_indices,
    extract_at_indices,
    prepare_data
)

__all__ = [
    'ERA5MSWX',
    'MSWXDWD',
    'ISIMIP_ERA5',
    'load_GHCN',
    'extract_dataset_metadata',
    'filter_ghcn_stations_by_metadata',
    'extract_model_at_stations',
    'assign_season',
    # Optimized extraction functions
    'build_kdtree',
    'query_station_indices',
    'extract_at_indices',
    'prepare_data'
]

__version__ = '0.3.0'  # Bumped for optimized functions
