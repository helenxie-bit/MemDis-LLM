# Run inference under case 1 (local memory)
numactl --cpunodebind=0 --membind=0 python inference.py --num_requests=30

# Run inference under case 2 (remote memory)
numactl --cpunodebind=0 python inference.py --kv_method=remote-memory --num_requests=30

# Run inference under case 3 (disk)
numactl --cpunodebind=0 python inference.py --kv_method=disk --num_requests=30

# Run inference under tiered memory
numactl --cpunodebind=0 python inference.py --tiered_kv_cache=True --num_requests=30