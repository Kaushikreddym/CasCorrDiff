#!/usr/bin/env python3
"""
Profile the __getitem__ method to find bottlenecks.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.mswxdwd import mswxdwd

def profile_getitem():
    """Profile individual steps in __getitem__."""
    
    print("Initializing dataset...")
    dataset = mswxdwd(
        data_path="/beegfs/muduchuru/data/",
        train=True,
        train_years=(1989, 2020),
        input_channels=["tas", "pr", "tasmax", "tasmin", "rsds"],
        output_channels=["tas", "pr", "tasmax", "tasmin", "hurs", "rsds"],
        static_channels=["elevation", "lsm", "dwd_mask", "pos_embed"],
        normalize=True,
        stats_dwd="/beegfs/muduchuru/data/hyras_stats_germany_log.json",
        stats_mswx="/beegfs/muduchuru/data/mswx_stats_germany_log.json",
    )
    
    print(f"\nProfiling data loading for 5 samples...\n")
    
    for idx in [0, 100, 500, 1000, 2000]:
        print(f"{'='*60}")
        print(f"Sample index: {idx}")
        print(f"{'='*60}")
        
        date = dataset.valid_dates[idx]
        print(f"Date: {date}")
        
        # Time MSWX loading
        t0 = time.time()
        arr_mswx = dataset._get_mswx(date)
        t_mswx = time.time() - t0
        print(f"  MSWX loading + regridding: {t_mswx:.3f}s")
        
        # Time HYRAS loading
        t0 = time.time()
        arr_dwd = dataset._get_dwd(date)
        t_dwd = time.time() - t0
        print(f"  HYRAS loading (+ rsds regrid): {t_dwd:.3f}s")
        
        # Time static channels (already loaded)
        t0 = time.time()
        if dataset.static_data is not None:
            import numpy as np
            arr_mswx_with_static = np.concatenate([arr_mswx, dataset.static_data], axis=0)
        else:
            arr_mswx_with_static = arr_mswx
        t_static = time.time() - t0
        print(f"  Static channels (concat): {t_static:.3f}s")
        
        # Time normalization
        t0 = time.time()
        _ = dataset.normalize_input(arr_mswx_with_static)
        t_norm_in = time.time() - t0
        
        t0 = time.time()
        _ = dataset.normalize_output(arr_dwd)
        t_norm_out = time.time() - t0
        print(f"  Normalization (in+out): {t_norm_in + t_norm_out:.3f}s")
        
        total = t_mswx + t_dwd + t_static + t_norm_in + t_norm_out
        print(f"  Total profiled: {total:.3f}s")
        print(f"\n  Breakdown:")
        print(f"    MSWX:   {t_mswx/total*100:.1f}%")
        print(f"    HYRAS:  {t_dwd/total*100:.1f}%")
        print(f"    Other:  {(t_static+t_norm_in+t_norm_out)/total*100:.1f}%")
        print()

if __name__ == "__main__":
    profile_getitem()
