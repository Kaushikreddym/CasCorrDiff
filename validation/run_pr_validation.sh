#!/bin/bash
# Run precipitation validation with sdba environment

cd /beegfs/muduchuru/codes/python/CasCorrDiff

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate sdba

# Run validation
python validation/validate_pr.py

echo "Precipitation validation complete!"
