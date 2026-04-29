#!/usr/bin/env python3
"""
Profile the entire training loop for a few samples.
"""

import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.mswxdwd import mswxdwd
from torch.utils.data import DataLoader
import torch.nn as nn

# Simple UNet-like model for profiling
class SimpleUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def profile_training():
    """Profile complete training iterations."""
    
    print("=" * 80)
    print("TRAINING PROFILING (2-3 samples)")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Initialize dataset
    print("\n[1] Initializing dataset...")
    t0 = time.time()
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
    print(f"   Dataset initialization: {time.time() - t0:.2f}s")
    
    # Create dataloader
    print("\n[2] Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,  # Single-threaded for clearer profiling
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Get sample to determine shapes
    sample_input, sample_output = dataset[0]
    in_channels = sample_input.shape[0]
    out_channels = sample_output.shape[0]
    
    print(f"   Input channels: {in_channels}")
    print(f"   Output channels: {out_channels}")
    
    # Initialize model
    print("\n[3] Initializing model...")
    t0 = time.time()
    model = SimpleUNet(in_channels, out_channels).to(device)
    print(f"   Model initialization: {time.time() - t0:.3f}s")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    
    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    print("\n[4] Running training iterations...")
    print("=" * 80)
    
    # Profile 3 iterations
    num_iterations = 3
    iter_times = []
    
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_iterations:
            break
        
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")
        iter_start = time.time()
        
        # Timing: Data loading (already done by dataloader)
        t0 = time.time()
        inputs = inputs.float().to(device)
        targets = targets.float().to(device)
        t_transfer = time.time() - t0
        print(f"  1. Data transfer to {device}: {t_transfer:.3f}s")
        
        # Timing: Forward pass
        t0 = time.time()
        optimizer.zero_grad()
        outputs = model(inputs)
        t_forward = time.time() - t0
        print(f"  2. Forward pass: {t_forward:.3f}s")
        
        # Timing: Loss computation
        t0 = time.time()
        loss = criterion(outputs, targets)
        t_loss = time.time() - t0
        print(f"  3. Loss computation: {t_loss:.3f}s")
        print(f"     Loss value: {loss.item():.6f}")
        
        # Timing: Backward pass
        t0 = time.time()
        loss.backward()
        t_backward = time.time() - t0
        print(f"  4. Backward pass: {t_backward:.3f}s")
        
        # Timing: Optimizer step
        t0 = time.time()
        optimizer.step()
        t_optim = time.time() - t0
        print(f"  5. Optimizer step: {t_optim:.3f}s")
        
        iter_time = time.time() - iter_start
        iter_times.append(iter_time)
        
        print(f"\n  Total iteration time: {iter_time:.3f}s")
        print(f"  Breakdown:")
        print(f"    Data transfer: {t_transfer/iter_time*100:5.1f}%")
        print(f"    Forward pass:  {t_forward/iter_time*100:5.1f}%")
        print(f"    Loss compute:  {t_loss/iter_time*100:5.1f}%")
        print(f"    Backward pass: {t_backward/iter_time*100:5.1f}%")
        print(f"    Optimizer:     {t_optim/iter_time*100:5.1f}%")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    import numpy as np
    iter_times = np.array(iter_times)
    print(f"Average iteration time: {iter_times.mean():.3f}s ± {iter_times.std():.3f}s")
    print(f"Samples/second: {2 / iter_times.mean():.2f}")  # batch_size=2
    
    # Estimate full training time
    total_samples = len(dataset)
    batch_size = 2
    batches_per_epoch = total_samples // batch_size
    time_per_epoch = batches_per_epoch * iter_times.mean()
    
    print(f"\nEstimated epoch time (single GPU, num_workers=0):")
    print(f"  {time_per_epoch / 60:.1f} minutes ({time_per_epoch / 3600:.2f} hours)")
    
    print("\nNOTE: This uses a simple model. Real training with larger models")
    print("      and num_workers > 0 will have different performance.")
    print("=" * 80)

if __name__ == "__main__":
    profile_training()
