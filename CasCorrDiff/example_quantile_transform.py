#!/usr/bin/env python3
"""
Example: Using Quantile Transform for Precipitation Normalization

This script demonstrates how to use the new quantile transform feature
for precipitation normalization in the MSWX-DWD dataset.
"""

import argparse
import torch
from torch.utils.data import DataLoader
from datasets.mswxdwd import mswxdwd


def main():
    parser = argparse.ArgumentParser(
        description="Example training script using quantile transform for precipitation"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/beegfs/muduchuru/data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--use-quantile-transform",
        action="store_true",
        help="Use quantile transform for precipitation normalization"
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["germany", "full"],
        default="germany",
        help="Spatial domain for statistics"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for DataLoader"
    )
    args = parser.parse_args()

    # Setup file paths based on domain
    domain_suffix = f"_{args.domain}" if args.domain != "full" else ""
    # Use standard scaler stats (no log or quantile transform)
    stats_mswx = f"{args.data_path}/mswx/mswx_stats.json"
    stats_dwd = f"{args.data_path}/HYRAS_DAILY/hyras_stats.json"
    
    qt_mswx = None
    qt_dwd = None
    if args.use_quantile_transform:
        qt_mswx = f"{args.data_path}/mswx_stats{domain_suffix}_quantile_transform.pkl"
        qt_dwd = f"{args.data_path}/hyras_stats{domain_suffix}_quantile_transform.pkl"

    print("="*70)
    print("MSWX-DWD Dataset Example with Quantile Transform")
    print("="*70)
    print(f"Domain: {args.domain}")
    print(f"Use quantile transform: {args.use_quantile_transform}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Initialize dataset with quantile transform
    print("Initializing dataset...")
    dataset = mswxdwd(
        data_path=args.data_path,
        train=True,
        input_channels=["pr", "tas", "hurs", "rsds"],
        output_channels=["pr", "tas", "tasmax", "tasmin"],
        static_channels=["elevation", "lsm", "dwd_mask"],
        normalize=True,
        stats_mswx=stats_mswx,
        stats_dwd=stats_dwd,
        quantile_transform_mswx=qt_mswx,
        quantile_transform_dwd=qt_dwd,
        use_quantile_transform=args.use_quantile_transform,
        patch_size=(128, 128)
    )
    print(f"✅ Dataset initialized with {len(dataset)} samples")
    print()

    # Create DataLoader
    print("Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    print(f"✅ DataLoader created")
    print()

    # Get a sample batch
    print("Loading first batch...")
    for batch_idx, (x, y) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        print(f"  Input channels: {x.shape[1]}")
        print(f"  Output channels: {y.shape[1]}")
        print()

        # Analyze normalization
        print("Normalization Analysis:")
        print(f"  Input - Min: {x.min():.4f}, Max: {x.max():.4f}, Mean: {x.mean():.4f}, Std: {x.std():.4f}")
        print(f"  Output - Min: {y.min():.4f}, Max: {y.max():.4f}, Mean: {y.mean():.4f}, Std: {y.std():.4f}")
        print()

        # Check precipitation channel (index 0)
        print("Precipitation Channel (pr) Analysis:")
        pr_input = x[:, 0]  # Batch dimension preserved
        pr_output = y[:, 0]
        print(f"  Input pr - Min: {pr_input.min():.4f}, Max: {pr_input.max():.4f}")
        print(f"  Output pr - Min: {pr_output.min():.4f}, Max: {pr_output.max():.4f}")
        
        if args.use_quantile_transform:
            print("  ✓ Using quantile transform (should be more uniform distribution)")
        else:
            print("  ✓ Using log-transform + z-score (standard method)")
        print()

        # Demonstrate denormalization
        print("Testing Denormalization...")
        y_denorm = dataset.denormalize_output(y.numpy())
        print(f"  Denormalized output shape: {y_denorm.shape}")
        print(f"  Denormalized pr - Min: {y_denorm[0, 0].min():.4f}, Max: {y_denorm[0, 0].max():.4f}")
        print()

        # Only process first batch for this example
        break

    print("="*70)
    print("Example completed successfully!")
    print("="*70)
    
    if args.use_quantile_transform:
        print("\n✅ Quantile Transform Summary:")
        print("  - Precipitation mapped to uniform [0, 1] during stats calculation")
        print("  - Scaled to [-1, 1] for network input")
        print("  - Inverse transform applied during denormalization")
        print("  - Other channels use standard z-score normalization")
    else:
        print("\n✅ Standard Log-Transform Summary:")
        print("  - Precipitation: log1p(x) then z-score normalization")
        print("  - Other channels: z-score normalization")


if __name__ == "__main__":
    main()
