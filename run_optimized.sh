#!/bin/bash

# Performance comparison script
# Run this to test the optimized pipeline

echo "=========================================="
echo "OPTIMIZED PIPELINE BENCHMARK"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Workers: 16 (4GB each)"
echo "  - Batch size: 50 stations/task"
echo "  - Years: 2020-2024"
echo ""
echo "Dashboard: http://localhost:8787"
echo ""
echo "Starting pipeline..."
echo ""

# Run with timing
time python extract_stations_parallel.py

echo ""
echo "=========================================="
echo "BENCHMARK COMPLETE"
echo "=========================================="
echo ""
echo "Check OPTIMIZATION_SUMMARY.md for details"
echo ""
