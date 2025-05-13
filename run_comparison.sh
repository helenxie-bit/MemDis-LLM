# Run inference under case 1 (local memory)
numactl --cpunodebind=0 --membind=0 python inference.py

# Run inference under case 2 (remote memory)
numactl --cpunodebind=0 --membind=1 python inference.py --kv_method=remote-memory

# Run inference under case 3 (disk)
numactl --cpunodebind=0 --membind=0 python inference.py --kv_method=disk
