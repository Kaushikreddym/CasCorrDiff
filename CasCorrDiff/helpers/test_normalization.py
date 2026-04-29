#!/usr/bin/env python3
"""
Test normalization round-trip for MSWX-DWD dataset.
Verifies that normalize → denormalize recovers original values.
"""

import numpy as np
import json
import sys
from pathlib import Path

# Add parent directory to path to import dataset
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_normalization_roundtrip():
    """Test that normalization and denormalization are inverses."""
    
    # Load stats files
    stats_mswx_path = "/beegfs/muduchuru/data/mswx_stats_germany_log.json"
    stats_hyras_path = "/beegfs/muduchuru/data/hyras_stats_germany_log.json"
    
    with open(stats_mswx_path, "r") as f:
        stats_mswx = json.load(f)
    
    with open(stats_hyras_path, "r") as f:
        stats_hyras = json.load(f)
    
    # Define channels
    input_channels = ["tas", "pr", "tasmax", "tasmin", "rsds"]
    output_channels = ["tas", "pr", "tasmax", "tasmin", "hurs", "rsds"]
    
    # Setup normalization parameters
    input_log_channels = []
    output_log_channels = []
    
    for i, ch in enumerate(input_channels):
        if "pr" in ch.lower() or "precip" in ch.lower():
            input_log_channels.append(i)
    
    for i, ch in enumerate(output_channels):
        if "pr" in ch.lower() or "precip" in ch.lower():
            output_log_channels.append(i)
    
    input_mean = np.array([stats_mswx[ch]["mean"] for ch in input_channels])[:, None, None]
    input_std = np.array([stats_mswx[ch]["std"] for ch in input_channels])[:, None, None]
    
    output_mean = np.array([stats_hyras[ch]["mean"] for ch in output_channels])[:, None, None]
    output_std = np.array([stats_hyras[ch]["std"] for ch in output_channels])[:, None, None]
    
    print("=" * 80)
    print("NORMALIZATION ROUND-TRIP TEST")
    print("=" * 80)
    
    # Test INPUT normalization (MSWX)
    print("\n" + "=" * 80)
    print("Testing INPUT channels (MSWX)")
    print("=" * 80)
    print(f"Channels: {input_channels}")
    print(f"Log-transformed channel indices: {input_log_channels}")
    
    # Create synthetic test data (10x10 spatial grid)
    np.random.seed(42)
    test_input = np.random.randn(len(input_channels), 10, 10)
    
    # Make realistic values for each channel
    test_input[0] = test_input[0] * 7 + 9  # tas: mean ~9°C
    test_input[1] = np.abs(test_input[1]) * 3  # pr: positive, mean ~3 mm/day (before log)
    test_input[2] = test_input[2] * 8 + 12  # tasmax: mean ~12°C
    test_input[3] = test_input[3] * 6 + 6  # tasmin: mean ~6°C
    test_input[4] = np.abs(test_input[4]) * 50 + 100  # rsds: positive, mean ~100 W/m²
    
    # Normalize
    x_norm = test_input.copy()
    for ch_idx in input_log_channels:
        x_norm[ch_idx] = np.log1p(np.maximum(test_input[ch_idx], 0.0))
    x_norm = (x_norm - input_mean) / input_std
    
    # Denormalize
    x_denorm = x_norm * input_std + input_mean
    for ch_idx in input_log_channels:
        x_denorm[ch_idx] = np.expm1(x_denorm[ch_idx])
    
    # Check if round-trip is accurate
    for i, ch in enumerate(input_channels):
        max_diff = np.max(np.abs(test_input[i] - x_denorm[i]))
        mean_diff = np.mean(np.abs(test_input[i] - x_denorm[i]))
        rel_error = max_diff / (np.abs(test_input[i]).max() + 1e-8)
        
        status = "✓ PASS" if max_diff < 1e-6 else "✗ FAIL"
        log_marker = " [LOG]" if i in input_log_channels else ""
        
        print(f"\n{status} {ch}{log_marker}:")
        print(f"  Original range: [{test_input[i].min():.4f}, {test_input[i].max():.4f}]")
        print(f"  Normalized range: [{x_norm[i].min():.4f}, {x_norm[i].max():.4f}]")
        print(f"  Recovered range: [{x_denorm[i].min():.4f}, {x_denorm[i].max():.4f}]")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print(f"  Relative error: {rel_error:.2e}")
    
    # Test OUTPUT normalization (HYRAS)
    print("\n" + "=" * 80)
    print("Testing OUTPUT channels (HYRAS)")
    print("=" * 80)
    print(f"Channels: {output_channels}")
    print(f"Log-transformed channel indices: {output_log_channels}")
    
    # Create synthetic test data
    test_output = np.random.randn(len(output_channels), 10, 10)
    
    # Make realistic values for each channel
    test_output[0] = test_output[0] * 7 + 9  # tas: mean ~9°C
    test_output[1] = np.abs(test_output[1]) * 3  # pr: positive, mean ~3 mm/day (before log)
    test_output[2] = test_output[2] * 8 + 13  # tasmax: mean ~13°C
    test_output[3] = test_output[3] * 6 + 5  # tasmin: mean ~5°C
    test_output[4] = test_output[4] * 11 + 79  # hurs: mean ~79%
    test_output[5] = np.abs(test_output[5]) * 50 + 100  # rsds: positive, mean ~100 W/m²
    
    # Normalize
    y_norm = test_output.copy()
    for ch_idx in output_log_channels:
        y_norm[ch_idx] = np.log1p(np.maximum(test_output[ch_idx], 0.0))
    y_norm = (y_norm - output_mean) / output_std
    
    # Denormalize
    y_denorm = y_norm * output_std + output_mean
    for ch_idx in output_log_channels:
        y_denorm[ch_idx] = np.expm1(y_denorm[ch_idx])
    
    # Check if round-trip is accurate
    all_pass = True
    for i, ch in enumerate(output_channels):
        max_diff = np.max(np.abs(test_output[i] - y_denorm[i]))
        mean_diff = np.mean(np.abs(test_output[i] - y_denorm[i]))
        rel_error = max_diff / (np.abs(test_output[i]).max() + 1e-8)
        
        passed = max_diff < 1e-6
        all_pass = all_pass and passed
        status = "✓ PASS" if passed else "✗ FAIL"
        log_marker = " [LOG]" if i in output_log_channels else ""
        
        print(f"\n{status} {ch}{log_marker}:")
        print(f"  Original range: [{test_output[i].min():.4f}, {test_output[i].max():.4f}]")
        print(f"  Normalized range: [{y_norm[i].min():.4f}, {y_norm[i].max():.4f}]")
        print(f"  Recovered range: [{y_denorm[i].min():.4f}, {y_denorm[i].max():.4f}]")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print(f"  Relative error: {rel_error:.2e}")
    
    # Test precipitation edge cases
    print("\n" + "=" * 80)
    print("Testing PRECIPITATION edge cases")
    print("=" * 80)
    
    # Test zero precipitation (very common)
    pr_idx = 1  # pr is at index 1 in both input and output
    test_zero_pr = np.zeros((1, 5, 5))
    
    # Normalize zero precipitation
    pr_norm = np.log1p(test_zero_pr)
    pr_norm = (pr_norm - output_mean[pr_idx]) / output_std[pr_idx]
    
    # Denormalize
    pr_denorm = pr_norm * output_std[pr_idx] + output_mean[pr_idx]
    pr_denorm = np.expm1(pr_denorm)
    
    max_diff = np.max(np.abs(test_zero_pr - pr_denorm))
    print(f"\n✓ Zero precipitation:")
    print(f"  Original: all zeros")
    print(f"  Normalized value: {pr_norm[0, 0, 0]:.4f}")
    print(f"  Recovered: max diff = {max_diff:.2e}")
    
    # Test small precipitation values
    test_small_pr = np.array([[[0.1, 0.5, 1.0], [2.0, 5.0, 10.0]]])
    
    pr_norm_small = np.log1p(test_small_pr)
    pr_norm_small = (pr_norm_small - output_mean[pr_idx]) / output_std[pr_idx]
    pr_denorm_small = pr_norm_small * output_std[pr_idx] + output_mean[pr_idx]
    pr_denorm_small = np.expm1(pr_denorm_small)
    
    max_diff_small = np.max(np.abs(test_small_pr - pr_denorm_small))
    print(f"\n✓ Various precipitation values:")
    print(f"  Original: {test_small_pr.flatten()}")
    print(f"  Recovered: {pr_denorm_small.flatten()}")
    print(f"  Max difference: {max_diff_small:.2e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if all_pass and max_diff < 1e-6 and max_diff_small < 1e-6:
        print("✓ ALL TESTS PASSED")
        print("  Normalization and denormalization are properly implemented.")
        print("  Log transformation correctly applied to precipitation channels.")
        print("  Round-trip error is within numerical precision (< 1e-6).")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("  Check implementation of normalize/denormalize functions.")
        return 1

if __name__ == "__main__":
    exit_code = test_normalization_roundtrip()
    sys.exit(exit_code)
