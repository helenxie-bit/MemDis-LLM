#!/bin/bash

set -e  # Exit on error

echo "=== Step 1: Installing system packages ==="
sudo apt update
sudo apt install -y numactl hwloc libnuma-dev

echo "=== Step 2: Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Step 3: Building Cython modules ==="
python setup.py build_ext --inplace

echo "=== Step 4: Creating results directory ==="
mkdir -p results

echo "=== Step 5: Generate workload ==="
python workloadGen.py --lambda_rate 5 --simulation_duration 50 --new_conv_prob 0.7 --seed 42

echo "=== Step 6: Running inference (local memory) ==="
numactl --cpunodebind=0 --membind=0 python inference.py

echo "=== Step 7: Running inference (remote memory) ==="
numactl --cpunodebind=0 python inference.py --kv_method=remote-memory

echo "=== Step 8: Running inference (disk) ==="
numactl --cpunodebind=0 python inference.py --kv_method=disk

echo "=== Step 9: Running inference (tiered memory) ==="
numactl --cpunodebind=0 python inference.py --tiered_kv_cache=True

echo "=== Step 9: Running inference (LRU Eviction) ==="
numactl --cpunodebind=0 python inference.py --kv_method=tiered-lru --lru_tiered_kv_cache=True --num_requests=30

echo "=== All experiments completed ==="
