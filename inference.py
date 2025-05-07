from contextlib import nullcontext
import os
import pandas as pd
import torch
import tiktoken
from model import GPT

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

# Generate text
with torch.no_grad():
    with ctx:
        for k in range(num_requests):
            y, updated_kv_cache, metrics, cpu_metrics = model.generate(
                x, 
                max_new_tokens=max_new_tokens,
                temperature=temperature, 
                top_k=top_k,
                kv_cache=total_kv_cache[k] if k in total_kv_cache else None
                )
            # print(decode(y[0].tolist()))
            # print("=" * 40)

            # Update KV cache
            total_kv_cache[k] = updated_kv_cache
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
            print(metrics_df)

            print(cpu_metrics)
