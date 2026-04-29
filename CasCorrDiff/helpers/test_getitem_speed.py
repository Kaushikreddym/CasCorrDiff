#!/usr/bin/env python3
"""
Test the speed of data loading with __getitem__ for MSWX-DWD dataset.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.mswxdwd import mswxdwd

def test_getitem_speed():
    """Test how long it takes to load samples."""
    
    print("=" * 80)
    print("TESTING DATA LOADING SPEED")
    print("=" * 80)
    
    # Initialize dataset
    print("\n[1] Initializing dataset...")
    init_start = time.time()
    
    dataset = mswxdwd(
        data_path="/beegfs/muduchuru/data/",
        train=True,
        train_years=(1989, 2020),
        val_years=(2021, 2024),
        input_channels=["tas", "pr", "tasmax", "tasmin", "rsds"],
        output_channels=["tas", "pr", "tasmax", "tasmin", "hurs", "rsds"],
        static_channels=["elevation", "lsm", "dwd_mask", "pos_embed"],
        normalize=True,
        stats_dwd="/beegfs/muduchuru/data/hyras_stats_germany_log.json",
        stats_mswx="/beegfs/muduchuru/data/mswx_stats_germany_log.json",
    )
    
    init_time = time.time() - init_start
    print(f"   Initialization took: {init_time:.2f} seconds")
    print(f"   Dataset length: {len(dataset)} samples")
    
    # Test single sample loading
    print("\n[2] Testing single sample loading...")
    
    num_samples = 10
    times = []
    
    for i in range(num_samples):
        idx = i * (len(dataset) // num_samples)  # Spread across dataset
        
        start = time.time()
        input_arr, output_arr = dataset[idx]
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"   Sample {i+1}/{num_samples} (idx={idx}): {elapsed:.3f}s "
              f"| Input shape: {input_arr.shape}, Output shape: {output_arr.shape}")
    
    # Statistics
    times = np.array(times)
    print("\n" + "=" * 80)
    print("TIMING STATISTICS")
    print("=" * 80)
    print(f"Mean time:   {times.mean():.3f} seconds")
    print(f"Median time: {np.median(times):.3f} seconds")
    print(f"Min time:    {times.min():.3f} seconds")
    print(f"Max time:    {times.max():.3f} seconds")
    print(f"Std dev:     {times.std():.3f} seconds")
    
    # Estimate throughput
    print("\n" + "=" * 80)
    print("THROUGHPUT ESTIMATES")
    print("=" * 80)
    samples_per_sec = 1.0 / times.mean()
    print(f"Samples per second: {samples_per_sec:.2f}")
    print(f"Samples per minute: {samples_per_sec * 60:.1f}")
    print(f"Samples per hour:   {samples_per_sec * 3600:.1f}")
    
    # Estimate epoch time
    batch_size = 8  # typical batch size
    num_workers = 4  # typical number of workers
    
    samples_per_sec_parallel = samples_per_sec * num_workers
    batches_per_epoch = len(dataset) // batch_size
    time_per_epoch_sec = batches_per_epoch / (samples_per_sec_parallel / batch_size)
    
    print(f"\nEstimated epoch time (batch_size={batch_size}, num_workers={num_workers}):")
    print(f"  {time_per_epoch_sec / 60:.1f} minutes ({time_per_epoch_sec / 3600:.2f} hours)")
    
    # Check data shapes
    print("\n" + "=" * 80)
    print("DATA DETAILS")
    print("=" * 80)
    sample_input, sample_output = dataset[0]
    print(f"Input channels:  {len(dataset.input_channels_list)} MSWX + {len(dataset.static_channels_list)} static")
    print(f"  MSWX channels: {dataset.input_channels_list}")
    print(f"  Static channels: {dataset.static_channels_list}")
    print(f"  Total input shape: {sample_input.shape}")
    print(f"\nOutput channels: {len(dataset.output_channels_list)} HYRAS")
    print(f"  Channels: {dataset.output_channels_list}")
    print(f"  Output shape: {sample_output.shape}")
    
    # Check for NaNs
    print(f"\nInput NaNs:  {np.isnan(sample_input).sum()} / {sample_input.size}")
    print(f"Output NaNs: {np.isnan(sample_output).sum()} / {sample_output.size}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_getitem_speed()
