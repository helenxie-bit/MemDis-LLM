from contextlib import nullcontext
import os
import numpy as np
import pandas as pd
import torch
import tiktoken
import time
from model import GPT
from kvDiskSim import save_kvcache_memmap, load_kvcache_memmap

# -----------------------------------------------------------
# Configuration
init_from = "gpt2" # Options: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
start = "\n" # Prompt to start text generation from (can also specify a file, use as: "FILE:prompt.txt")
num_requests = 5
max_new_tokens = 100
temperature = 0.0 # In order to get deterministic results for reproduction, set temperature to 0.0
top_k = 200 # Retain only the top k tokens with highest probability (not used if temperature==0.0)
seed = 42 # Random seed for reproducibility
device = "cpu" # Options: "cpu", "cuda" (if available)
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
metrics_file = "results/metrics.csv" # File to save metrics
cpu_metrics_file = "results/cpu_clock_metrics.csv" # File to save metrics
kv_method = "memory" # Options: "memory", "disk"
kv_cache_dir = './kv_cache_disk/'
if kv_method == 'disk':
    os.makedirs(kv_cache_dir, exist_ok=True)

exec(open("configurator.py").read()) # Overrides from command line or config file
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
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

total_kv_cache = {} # Dictionary to store KV cache for each request

gen_count = 0
generation_cycle_times = []
# Generate text
with torch.no_grad():
    with ctx:
        for k in range(num_requests):
            # Measuring NUMA performace: we measure wall clock time between each generate() function to see if clock times went down
            gen_start_time = time.perf_counter()
            #if kv_method == 'memory':
            kv_cache = total_kv_cache.get(k, None)
            #else:
            #    kv_cache = load_kvcache_memmap(k, kv_cache_dir, device)
            y, updated_kv_cache, metrics = model.generate(
                x, 
                max_new_tokens=max_new_tokens,
                temperature=temperature, 
                top_k=top_k,
                kv_cache=kv_cache,
                kv_method=kv_method,
                kv_cache_dir=kv_cache_dir,
                device=device,
                request_id=k,
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

            # Update KV cache
            #if kv_method == 'memory':
            total_kv_cache[k] = updated_kv_cache
            #else:
            #    save_kvcache_memmap(k, updated_kv_cache, kv_cache_dir)
            total_bytes = sum(
                keys.element_size() * keys.numel() + values.element_size() * values.numel()
                for tensor_list in total_kv_cache.values()
                for keys, values in tensor_list
            )
            print(f"Total KV cache size after {k}th request: {total_bytes / (1024 ** 2):.2f} MB")

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
