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

echo "=== Step 5: Running inference (local memory) ==="
numactl --cpunodebind=0 --membind=0 python inference.py --num_requests=30

echo "=== Step 6: Running inference (remote memory) ==="
numactl --cpunodebind=0 python inference.py --kv_method=remote-memory --num_requests=30

echo "=== Step 7: Running inference (disk) ==="
numactl --cpunodebind=0 python inference.py --kv_method=disk --num_requests=30

echo "=== Step 8: Running inference (tiered memory) ==="
numactl --cpunodebind=0 python inference.py --tiered_kv_cache=True --num_requests=30

echo "=== All experiments completed ==="
