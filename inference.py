from contextlib import nullcontext
import os
import numpy as np
import pandas as pd
import torch
import tiktoken
import time
from model import GPT
from kvDiskSim import get_dir_size
from kvRemoteSim import get_remote_kvcache_memory_usage

# -----------------------------------------------------------
# Configuration
init_from = "gpt2" # Options: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
start = "FILE:data/input.txt" # Prompt to start text generation from (can also specify a file, use as: "FILE:prompt.txt")
num_requests = 10
input_tokens = 1000
max_new_tokens = 20
temperature = 0.0 # In order to get deterministic results for reproduction, set temperature to 0.0
top_k = 200 # Retain only the top k tokens with highest probability (not used if temperature==0.0)
seed = 42 # Random seed for reproducibility
device = "cpu" # Options: "cpu", "cuda" (if available)
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
kv_method = "local-memory" # Options: "local-memory", "remote-memory", "disk"
kv_cache_dir = './kv_cache_disk/'
if kv_method == 'disk':
    os.makedirs(kv_cache_dir, exist_ok=True)
tiered_kv_cache = False # Whether to use tiered KV cache
memory_limit = 1024 # Configure according to your system, here we set it to 1 GB
memory_threshold = 0.7 # Memory threshold for switching to next tier
remote_node = 1 # NUMA node to allocate on (if using remote memory)

exec(open("configurator.py").read()) # Overrides from command line or config file

metrics_file = f"results/metrics_{kv_method}.csv" # File to save metrics
cpu_metrics_file = f"results/cpu_clock_metrics_{kv_method}.csv" # File to save metrics
# -----------------------------------------------------------

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

gen_count = 0
generation_cycle_times = []
# Generate text
with torch.no_grad():
    with ctx:
        for k in range(num_requests):
            # Measuring NUMA performace: we measure wall clock time between each generate() function to see if clock times went down
            gen_start_time = time.perf_counter()

            y, updated_kv_cache, metrics = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                kv_method=kv_method,
                kv_cache=(
                    total_kv_cache_local.get(k, None) if kv_method == "local-memory"
                    else total_kv_cache_remote.get(k, None) if kv_method == "remote-memory"
                    else None
                ),
                request_id=k,
                remote_memory_var=total_kv_cache_remote if kv_method == "remote-memory" else None,
                remote_node=remote_node,
                kv_cache_dir=kv_cache_dir,
                device=device,
            )
            # print(decode(y[0].tolist()))
            # print("=" * 40)

            # Take note of how much time it took
            gen_end_time = time.perf_counter()
            elapsed_time = gen_end_time - gen_start_time

            generation_cycle_times.append({
                "generation_index": gen_count,
                "elapsed_time_seconds": elapsed_time
            })
            gen_count += 1

            # Update the dictionary which stores KV cache if using memory method
            if kv_method == "local-memory":
                total_kv_cache_local[k] = updated_kv_cache
                kv_cache_size = sum(
                    keys.element_size() * keys.numel()
                    + values.element_size() * values.numel()
                    for tensor_list in total_kv_cache_local.values()
                    for keys, values in tensor_list
                ) / (1024 ** 2)  # Convert to MB
                print(f"Total KV cache size after {k}th request: {kv_cache_size:.2f} MB")

                if tiered_kv_cache == True and kv_cache_size > memory_limit * memory_threshold:
                    print(f"Warning: Memory usage exceeded threshold, switching to remote memory...")
                    kv_method = "remote-memory"
                    total_kv_cache_remote = {}  # Initialize remote cache in this case

            elif kv_method == "remote-memory":
                kv_cache_size = get_remote_kvcache_memory_usage(total_kv_cache_remote) / (1024 ** 2)  # Convert to MB
                print(f"Total KV cache size after {k}th request: {kv_cache_size:.2f} MB")

                if tiered_kv_cache ==True and kv_cache_size > memory_limit * memory_threshold:
                    print(f"Warning: Memory usage exceeded threshold, switching to disk...")
                    kv_method = "disk"
                    os.makedirs(kv_cache_dir, exist_ok=True) # Initialize the directory to store cache in disk in this case

            else:
                kv_cache_size = get_dir_size(kv_cache_dir) / (1024 ** 2)  # Convert to MB
                print(f"Total KV cache size after {k}th request: {kv_cache_size:.2f} MB")

            # Save metrics to a DataFrame and a CSV file
            metrics["model"] = init_from
            metrics_df = pd.DataFrame([metrics])
            metrics_file_exists = os.path.exists(metrics_file)
            metrics_df.to_csv(metrics_file, mode='a', header=not metrics_file_exists, index=False) 
            # print(metrics_df)

    cycle_times = pd.DataFrame(generation_cycle_times)
    cycle_times.to_csv(cpu_metrics_file, index=False)
    print(generation_cycle_times)

# kv-cache cleanup
if kv_method == "disk" and os.path.isdir(kv_cache_dir) and "kv_cache_disk" in kv_cache_dir:
    for root, _, files in os.walk(kv_cache_dir, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        os.rmdir(root)
