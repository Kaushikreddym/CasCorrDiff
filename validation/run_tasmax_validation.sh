#!/bin/bash
# Run tasmax validation with sdba environment

cd /beegfs/muduchuru/codes/python/CasCorrDiff

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate sdba

# Run validation
python validation/validate_tasmax.py

echo "Temperature validation complete!"
