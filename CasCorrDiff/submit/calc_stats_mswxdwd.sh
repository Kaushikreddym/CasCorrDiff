#!/bin/bash
#SBATCH --job-name=calc_stats             # Job name
#SBATCH --output=submit/logs/calc_stats_mswxdwd.%j.out  # STDOUT file
#SBATCH --error=submit/logs/calc_stats_mswxdwd.%j.err   # STDERR file
#SBATCH --time=2-00:00:00                 # Runtime (D-HH:MM:SS) - 2 days for full dataset
#SBATCH --ntasks=1                        # Single task
#SBATCH --cpus-per-task=80                # CPU cores for parallel processing
#SBATCH --partition=highmem                   # Fat node partition for high memory/CPU
#SBATCH --mem=180G                        # Memory allocation
##SBATCH --hint=nomultithread             # Optional: for CPU affinity

# --- Load environment ---
source ~/.bashrc
conda activate sdba

# --- Move to project directory ---
cd /beegfs/muduchuru/codes/physicsnemo/examples/weather/corrdiff || exit 1

# --- Print system info ---
echo "Running on node: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# --- Run the statistics calculation script ---
# Computes log-transformed normalization statistics for MSWX and HYRAS datasets
# Germany domain: lat [47, 55], lon [6, 15]
# Output files: /beegfs/muduchuru/data/mswx_stats_germany_log.json
#               /beegfs/muduchuru/data/hyras_stats_germany_log.json

python -u helpers/calc_stats_mswxdwd.py \
    --domain germany \
    --mswx-path /beegfs/muduchuru/data/mswx \
    --hyras-path /beegfs/muduchuru/data/HYRAS_DAILY \
    --output-dir /beegfs/muduchuru/data \
    --skip-hyras
    --n-workers 80
# Uncomment to test with limited files first:
# python helpers/calc_stats_mswxdwd.py --domain germany --max-files 100

# Uncomment for full spatial extent instead of Germany:
# python helpers/calc_stats_mswxdwd.py --domain full

echo ""
echo "Job completed at: $(date)"
