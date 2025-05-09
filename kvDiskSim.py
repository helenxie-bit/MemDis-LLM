import os
import numpy as np
import torch
import json

def save_kvcache_memmap(request_id, updated_kv_cache, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    metadata = {}

    for i, (k_tensor, v_tensor) in enumerate(updated_kv_cache):
        # File paths
        k_path = os.path.join(save_dir, f"req{request_id}_layer{i}_key.dat")
        v_path = os.path.join(save_dir, f"req{request_id}_layer{i}_value.dat")

        # Convert to numpy and save raw binary using tofile()
        k_np = k_tensor.detach().cpu().numpy()
        v_np = v_tensor.detach().cpu().numpy()

        k_np.tofile(k_path)
        v_np.tofile(v_path)

        # Save metadata: shapes + dtype
        metadata[f"layer_{i}"] = {
            "k_shape": list(k_np.shape),
            "v_shape": list(v_np.shape),
            "dtype": str(k_np.dtype)
        }

    # Save JSON metadata
    meta_path = os.path.join(save_dir, f"req{request_id}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)


def load_kvcache_memmap(request_id, save_dir, device):
    meta_path = os.path.join(save_dir, f"req{request_id}_meta.json")
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    kv_cache = []
    for i in range(len(metadata)):
        meta = metadata[f"layer_{i}"]
        dtype = np.dtype(meta["dtype"])
        k_shape = tuple(meta["k_shape"])
        v_shape = tuple(meta["v_shape"])

        k_path = os.path.join(save_dir, f"req{request_id}_layer{i}_key.dat")
        v_path = os.path.join(save_dir, f"req{request_id}_layer{i}_value.dat")

        # Map files into memory (lazy load)
        k_mem = np.memmap(k_path, dtype=dtype, mode='r', shape=k_shape)
        v_mem = np.memmap(v_path, dtype=dtype, mode='r', shape=v_shape)

        k_tensor = torch.from_numpy(k_mem).to(device)
        v_tensor = torch.from_numpy(v_mem).to(device)

        kv_cache.append((k_tensor, v_tensor))

    return kv_cache
