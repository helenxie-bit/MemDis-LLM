# Install packages needed for NUMA node
sudo apt update
sudo apt install -y numactl hwloc
sudo apt install libnuma-dev

# Install dependencies
pip install -r requirements.txt

# Create a directory for the results
mkdir -p results

# Run inference under case 1 (local memory)
numactl --cpunodebind=0 --membind=0 python inference.py

# Run inference under case 2 (remote memory)
numactl --cpunodebind=0 python inference.py --kv_method=remote-memory --numa_node=1

# Run inference under case 3 (disk)
numactl --cpunodebind=0 python inference.py --kv_method=disk

# Run inference under tiered memory
numactl --cpunodebind=0 python inference.py --tiered_kv_cache=True --num_requests=50