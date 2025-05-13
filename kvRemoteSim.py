import torch
import numa_bind

# Helper: Allocate on remote NUMA node
def allocate_on_remote_numa(shape, dtype=torch.float16, local_node=0, remote_node=1):
    numa_bind.set_membind(remote_node)
    tensor = torch.empty(shape, dtype=dtype)
    tensor.fill_(0)  # Initialize to zero
    numa_bind.set_membind(local_node)  # Reset to default NUMA node
    return tensor

# Save KV Cache to Remote NUMA Node
def save_kvcache_remote(request_id, layer_id, updated_kv_cache, kv_cache_store, local_node=0, remote_node=1):
    k_tensor, v_tensor = updated_kv_cache

    # Copy tensors to remote NUMA memory
    k_remote = allocate_on_remote_numa(k_tensor.shape, k_tensor.dtype, local_node, remote_node)
    v_remote = allocate_on_remote_numa(v_tensor.shape, v_tensor.dtype, local_node, remote_node)

    k_remote.copy_(k_tensor)
    v_remote.copy_(v_tensor)

    if request_id not in kv_cache_store:
        kv_cache_store[request_id] = {}

    kv_cache_store[request_id][layer_id] = (k_remote, v_remote)

# Load KV Cache from Remote NUMA Node
def load_kvcache_remote(request_id, layer_id, kv_cache_store):
    if request_id in kv_cache_store and layer_id in kv_cache_store[request_id]:
        return kv_cache_store[request_id][layer_id]
    return None

# Calculate Total Memory Usage (Bytes)
def get_remote_kvcache_memory_usage(kv_cache_store):
    total_bytes = 0
    for req_id in kv_cache_store:
        for layer_id in kv_cache_store[req_id]:
            k, v = kv_cache_store[req_id][layer_id]
            total_bytes += k.element_size() * k.numel()
            total_bytes += v.element_size() * v.numel()
    return total_bytes
