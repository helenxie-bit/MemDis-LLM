from contextlib import nullcontext
import os
import numpy as np
import pandas as pd
import torch
import tiktoken
import time
import random  # Add import for random eviction
from model import GPT
from kvDiskSim import get_dir_size
import numa_bind
import psutil
import json
from tiered_kv_cache import LRUTieredKVCache
from memoryMonitor import get_numastat

# -----------------------------------------------------------
# Configuration
init_from = "gpt2" # Options: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
start = "FILE:data/input.txt" # Prompt to start text generation from (can also specify a file, use as: "FILE:prompt.txt")
input_tokens = 500
max_new_tokens = 20
temperature = 0.0 # In order to get deterministic results for reproduction, set temperature to 0.0
top_k = 200 # Retain only the top k tokens with highest probability (not used if temperature==0.0)
seed = 42 # Random seed for reproducibility
device = "cpu" # Options: "cpu", "cuda" (if available)
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
kv_method = "local-memory" # Options: "local-memory", "remote-memory", "disk", "tiered-lru"
kv_cache_dir = './kv_cache_disk/'
if kv_method == 'disk':
    os.makedirs(kv_cache_dir, exist_ok=True)
tiered_kv_cache = False # Whether to use naive tiered KV cache
lru_tiered_kv_cache = False # Whether to use LRU tiered KV cache
memory_limit = 1024 # Configure according to your system, here we set it to 1 GB
memory_threshold = 0.7 # Memory threshold for switching to next tier
local_node = 0
remote_node = 1 # NUMA node to allocate on (if using remote memory)

# LRU Tiered cache configuration
lru_local_limit_mb = 1024  # Local memory limit in MB
lru_remote_limit_mb = 1024  # Remote memory limit in MB
lru_local_threshold = 0.7  # Local memory threshold for eviction
lru_remote_threshold = 0.7  # Remote memory threshold for eviction

exec(open("configurator.py").read()) # Overrides from command line or config file

# Initialize LRU tiered cache if requested
tiered_cache_manager = None
if lru_tiered_kv_cache or kv_method == "tiered-lru":
    tiered_cache_manager = LRUTieredKVCache(
        local_limit_mb=lru_local_limit_mb,
        remote_limit_mb=lru_remote_limit_mb,
        local_threshold=lru_local_threshold,
        remote_threshold=lru_remote_threshold,
        local_node=local_node,
        remote_node=remote_node,
        kv_cache_dir=kv_cache_dir
    )
    kv_method = "tiered-lru"

metrics_file = f"results/metrics_{lru_tiered_kv_cache if lru_tiered_kv_cache else tiered_kv_cache}_{kv_method}.csv"
cpu_metrics_file = f"results/cpu_clock_metrics_{lru_tiered_kv_cache if lru_tiered_kv_cache else tiered_kv_cache}_{kv_method}.csv"

# -----------------------------------------------------------
if kv_method == "remote-memory":
    numa_bind.set_membind(remote_node)  # Set memory binding to remote NUMA node

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
ptdtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=ptdtype)

# Load pretrained model
model = GPT.from_pretrained(init_from, dict(dropout=0.0))
model.eval()
model.to(device)

# Tokenization
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# Encode the prompt
if start.startswith("FILE:"):
    with open(start[5:], "r") as f:
        start = f.read()
start_ids = encode(start)
x = torch.tensor(start_ids[:input_tokens], dtype=torch.long, device=device)[None, ...]

# Initialize KV cache
if kv_method == "local-memory":
    total_kv_cache_local = {}  # Dictionary to store KV cache for each request if using local memory
elif kv_method == "remote-memory":    
    total_kv_cache_remote = {} # Dictionary to store KV cache for each request if using remote memory
else:
    os.makedirs(kv_cache_dir, exist_ok=True)  # Directory to store KV cache files if using disk
kv_cache_size_local, kv_cache_size_remote, kv_cache_size_disk = 0, 0, 0

# Read workload
with open("workload.json", "r") as f:
    requests = json.load(f)

gen_count = 0
generation_cycle_times = []

# Get the pid so that the psutil can start monitoring
process = psutil.Process(os.getpid())
# Get initial memory usage
init_total_memory = get_numastat(process.pid)["Total_MB"]
used_local_memory, used_remote_memory = 0, 0

processed_requests = []
start_time = time.perf_counter() # Reference time
# Generate text
with torch.no_grad():
    with ctx:
        #for k in range(num_requests):
        for request in requests:
            # Measuring NUMA performace: we measure wall clock time between each generate() function to see if clock times went down
            gen_start_time = time.perf_counter()

            # Synchronize to match Poisson arrival times
            now = time.perf_counter()
            wait_time = request["arrival_time"] - (now - start_time)
            if wait_time > 0:
                time.sleep(wait_time)
            
            request_id = request["request_id"]
            if request_id in processed_requests:
                is_old_conversation = True
            else:
                is_old_conversation = False
                processed_requests.append(request_id)
            print(f"📌 Request ID: {request_id}, Old Conversation: {is_old_conversation}")

            y, updated_kv_cache, metrics = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                kv_method=kv_method,
                kv_cache=(
                    total_kv_cache_local.get(request_id, None) if kv_method == "local-memory"
                    else total_kv_cache_remote.get(request_id, None) if kv_method == "remote-memory"
                    else None
                ),
                request_id=request_id,
                kv_cache_dir=kv_cache_dir,
                device=device,
                is_old_conversation=is_old_conversation,
                tiered_cache_manager=tiered_cache_manager,  # Pass the LRU cache manager
            )

            # Take note of how much time it took
            gen_end_time = time.perf_counter()
            elapsed_time = gen_end_time - gen_start_time

            generation_cycle_times.append({
                "generation_index": gen_count,
                "elapsed_time_seconds": elapsed_time
            })
            gen_count += 1

            # Handle different cache methods
            if kv_method == "tiered-lru":
                # Print cache statistics
                stats = tiered_cache_manager.get_stats()
                print(f"Request {request_id} - Cache Stats:")
                print(f"  Local: {stats['local_size_mb']:.2f}MB ({stats['local_count']} items, {stats['local_utilization']:.1%} util)")
                print(f"  Remote: {stats['remote_size_mb']:.2f}MB ({stats['remote_count']} items, {stats['remote_utilization']:.1%} util)")
                print(f"  Disk: {stats['disk_count']} items")
                
                total_size_mb = stats['local_size_mb'] + stats['remote_size_mb']
                print(f"Total KV cache size after {request_id}th request: {total_size_mb:.2f} MB")

            elif kv_method == "local-memory":
                total_kv_cache_local[request_id] = updated_kv_cache
                kv_cache_size_local = sum(
                    keys.element_size() * keys.numel()
                    + values.element_size() * values.numel()
                    for tensor_list in total_kv_cache_local.values()
                    for keys, values in tensor_list
                ) / (1024 ** 2)  # Convert to MB
                
                # Check if memory limit exceeded and remove random entries if tiered_kv_cache is False
                if not tiered_kv_cache and kv_cache_size_local >= memory_limit * memory_threshold:
                    print(f"Memory limit exceeded, removing random entries...")
                    cache_keys = list(total_kv_cache_local.keys())
                    
                    while kv_cache_size_local >= memory_limit * memory_threshold and cache_keys:
                        # Randomly select a key to evict
                        evict_key = random.choice(cache_keys)
                        cache_keys.remove(evict_key)
                        
                        # Remove the entry
                        del total_kv_cache_local[evict_key]
                        
                        # Recalculate cache size
                        kv_cache_size_local = sum(
                            keys.element_size() * keys.numel()
                            + values.element_size() * values.numel()
                            for tensor_list in total_kv_cache_local.values()
                            for keys, values in tensor_list
                        ) / (1024 ** 2)
                
                print(f"Total KV cache size after {request_id}th request: {kv_cache_size_local:.2f} MB")

                curr_total_memory = get_numastat(process.pid)["Total_MB"]
                used_local_memory = curr_total_memory - init_total_memory
                print(f"Used local memory: {used_local_memory:.2f} MB")

                if tiered_kv_cache == True and used_local_memory >= memory_limit * memory_threshold:
                    print(f"Warning: Memory usage exceeded threshold, switching to remote memory...")
                    kv_method = "remote-memory"
                    numa_bind.set_membind(remote_node)  # Set memory binding to remote NUMA node
                    total_kv_cache_remote = {}  # Initialize remote cache in this case

            elif kv_method == "remote-memory":
                total_kv_cache_remote[request_id] = updated_kv_cache
                kv_cache_size_remote = sum(
                    keys.element_size() * keys.numel()
                    + values.element_size() * values.numel()
                    for tensor_list in total_kv_cache_remote.values()
                    for keys, values in tensor_list
                ) / (1024 ** 2)
                
                # Check if memory limit exceeded and remove random entries if tiered_kv_cache is False
                if not tiered_kv_cache and kv_cache_size_remote >= memory_limit * memory_threshold:
                    print(f"Memory limit exceeded, removing random entries...")
                    cache_keys = list(total_kv_cache_remote.keys())
                    
                    while kv_cache_size_remote >= memory_limit * memory_threshold and cache_keys:
                        # Randomly select a key to evict
                        evict_key = random.choice(cache_keys)
                        cache_keys.remove(evict_key)
                        
                        # Remove the entry
                        del total_kv_cache_remote[evict_key]
                        
                        # Recalculate cache size
                        kv_cache_size_remote = sum(
                            keys.element_size() * keys.numel()
                            + values.element_size() * values.numel()
                            for tensor_list in total_kv_cache_remote.values()
                            for keys, values in tensor_list
                        ) / (1024 ** 2)
                
                print(f"Total KV cache size after {request_id}th request: {kv_cache_size_local + kv_cache_size_remote:.2f} MB")

                curr_total_memory = get_numastat(process.pid)["Total_MB"]
                used_remote_memory = curr_total_memory - init_total_memory - used_local_memory
                print(f"Used remote memory: {used_remote_memory:.2f} MB")

                if tiered_kv_cache ==True and used_remote_memory >= memory_limit * memory_threshold:
                    print(f"Warning: Memory usage exceeded threshold, switching to disk...")
                    kv_method = "disk"
                    numa_bind.set_membind(local_node)  # Set memory binding to local NUMA node
                    os.makedirs(kv_cache_dir, exist_ok=True) # Initialize the directory to store cache in disk in this case

            else:
                kv_cache_size_disk = get_dir_size(kv_cache_dir) / (1024 ** 2)  # Convert to MB
                print(f"Total KV cache size after {request_id}th request: {kv_cache_size_local + kv_cache_size_remote + kv_cache_size_disk:.2f} MB")

            # Save metrics to a DataFrame and a CSV file
            metrics["model"] = init_from
            
            # Add cache statistics for LRU tiered cache
            if kv_method == "tiered-lru":
                stats = tiered_cache_manager.get_stats()
                metrics.update({
                    "local_cache_size_mb": stats['local_size_mb'],
                    "remote_cache_size_mb": stats['remote_size_mb'],
                    "disk_cache_count": stats['disk_count'],
                    "local_cache_utilization": stats['local_utilization'],
                    "remote_cache_utilization": stats['remote_utilization'],
                })
            
            metrics_df = pd.DataFrame([metrics])
            metrics_file_exists = os.path.exists(metrics_file)
            metrics_df.to_csv(metrics_file, mode='a', header=not metrics_file_exists, index=False)
            # print(metrics_df)

    cycle_times = pd.DataFrame(generation_cycle_times)
    cycle_times.to_csv(cpu_metrics_file, index=False)
    #print(generation_cycle_times)

# Cleanup
if tiered_cache_manager:
    tiered_cache_manager.cleanup()
elif kv_method == "disk" and os.path.isdir(kv_cache_dir) and "kv_cache_disk" in kv_cache_dir:
    for root, _, files in os.walk(kv_cache_dir, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        os.rmdir(root)
