from contextlib import nullcontext
import os
import numpy as np
import pandas as pd
import torch
import tiktoken
import time
from model import GPT
from kvDiskSim import get_dir_size
import numa_bind
import psutil
import subprocess

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
local_node = 0
remote_node = 1 # NUMA node to allocate on (if using remote memory)

exec(open("configurator.py").read()) # Overrides from command line or config file


metrics_file = f"results/metrics_{tiered_kv_cache}_{kv_method}.csv" # File to save metrics
cpu_metrics_file = f"results/cpu_clock_metrics_{tiered_kv_cache}_{kv_method}.csv" # File to save metrics
# -----------------------------------------------------------
if kv_method == "remote-memory":
    numa_bind.set_membind(remote_node)  # Set memory binding to remote NUMA node

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
ptdtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=ptdtype)

def get_numastat(pid):
    try:
        result = subprocess.run(["numastat", "-p", str(pid)], capture_output=True, text=True, check=True)
        #print(f"NUMA statistics for PID {pid}:\n{result.stdout}")
        lines = result.stdout.splitlines()
        for line in lines:
            if line.strip().startswith("Total"):
                parts = line.split()
                # Assuming format: 'Total', <Node0>, <Node1>, <Total>
                return {
                    "Node0_MB": float(parts[1]),
                    "Node1_MB": float(parts[2]),
                    "Total_MB": float(parts[3])
                }

        print("Could not find 'Total' line in numastat output.")
        return {}
    except Exception as e:
        print(f"Error reading numastat: {e}")
        return {}

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

gen_count = 0
generation_cycle_times = []

# Get the pid so that the psutil can start monitoring
process = psutil.Process(os.getpid())

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
            # if kv_method == "local-memory":
            #     total_kv_cache_local[k] = updated_kv_cache
            #     kv_cache_size_local = sum(
            #         keys.element_size() * keys.numel()
            #         + values.element_size() * values.numel()
            #         for tensor_list in total_kv_cache_local.values()
            #         for keys, values in tensor_list
            #     ) / (1024 ** 2)  # Convert to MB
            #     print(f"Total KV cache size after {k}th request: {kv_cache_size_local:.2f} MB")

            #     if tiered_kv_cache == True and kv_cache_size_local >= memory_limit * memory_threshold:
            #         print(f"Warning: Memory usage exceeded threshold, switching to remote memory...")
            #         kv_method = "remote-memory"
            #         numa_bind.set_membind(remote_node)  # Set memory binding to remote NUMA node
            #         total_kv_cache_remote = {}  # Initialize remote cache in this case

            if kv_method == "remote-memory":
                total_kv_cache_remote[k] = updated_kv_cache
                kv_cache_size_remote = sum(
                    keys.element_size() * keys.numel()
                    + values.element_size() * values.numel()
                    for tensor_list in total_kv_cache_remote.values()
                    for keys, values in tensor_list
                ) / (1024 ** 2)
                print(f"Total KV cache size after {k}th request: {kv_cache_size_local + kv_cache_size_remote:.2f} MB")

                if tiered_kv_cache ==True and kv_cache_size_remote >= memory_limit * memory_threshold:
                    #print(f"Warning: Memory usage exceeded threshold, switching to disk...")
                    print(f"Warning: Memory usage exceeded threshold, switching to local memory...")
                    #kv_method = "disk"
                    kv_method = "local-memory"
                    numa_bind.set_membind(local_node)  # Set memory binding to local NUMA node
                    os.makedirs(kv_cache_dir, exist_ok=True) # Initialize the directory to store cache in disk in this case
            
            elif kv_method == "local-memory":
                total_kv_cache_local[k] = updated_kv_cache
                kv_cache_size_local = sum(
                    keys.element_size() * keys.numel()
                    + values.element_size() * values.numel()
                    for tensor_list in total_kv_cache_local.values()
                    for keys, values in tensor_list
                ) / (1024 ** 2)  # Convert to MB
                print(f"Total KV cache size after {k}th request: {kv_cache_size_local:.2f} MB")

                if tiered_kv_cache == True and kv_cache_size_local >= memory_limit * memory_threshold:
                    print(f"Warning: Memory usage exceeded threshold, switching to disk...")
                    kv_method = "disk"
                    #numa_bind.set_membind(remote_node)  # Set memory binding to remote NUMA node
                    total_kv_cache_remote = {}  # Initialize remote cache in this case

            else:
                kv_cache_size_disk = get_dir_size(kv_cache_dir) / (1024 ** 2)  # Convert to MB
                print(f"Total KV cache size after {k}th request: {kv_cache_size_local + kv_cache_size_remote + kv_cache_size_disk:.2f} MB")

            # --- Code for getting memory usage --- #
            try:
                # Get various data on memory usage
                memory_info = process.memory_info()

                # RSS: the amount the physical memory thjat the process is currently using (amount that is not swapperd to disk)
                metrics["process_memory_rss_mb"] = memory_info.rss / (1024 * 1024)

               # VMS: Virtual memory, which includes disk swap usage, RAM usage, etc.
                metrics["process_memory_vms_mb"] = memory_info.vms / (1024 * 1024)


            except psutil.AccessDenied:
                print(f"Warning: Could not access process memory info (AccessDenied)")
                metrics["process_memory_rss_mb"] = np.nan
                # metrics["process_memory_vms_mb"] = np.nan
                # metrics["process_memory_uss_mb"] = np.nan
            except Exception as e:
                print(f"Warning: Could not access process memory info: {e}")
                metrics["process_memory_rss_mb"] = np.nan
            
            pid = process.pid
            #print(f"Process ID: {pid}")
            metrics["numa_memory_usage"] = get_numastat(pid)

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
