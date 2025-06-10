# MemDis-LLM

**MemDis-LLM** explores the use of **memory disaggregation** in large language model (LLM) inference by implementing a **tiered memory system**. When local memory is insufficient, this system prioritizes disaggregated memory over slower disk-based offloading, aiming to improve performance under memory constraints.

---

## 💼 Project Structure Overview

## 💼 Project Structure Overview

This project is built on top of [NanoGPT](https://github.com/karpathy/nanoGPT) and extends it to support memory disaggregation and tiered KV cache strategies. The codebase is organized as follows:

### Core Components
- `workloadGen.py` – Generates synthetic inference workloads to simulate different usage patterns
- `inference.py` – Main entry point for running GPT-2 inference with configurable memory strategies
- `tiered_kv_cache.py` – Implements tiered KV cache logic, including LRU-based cache replacement
- `model.py` – Handles GPT-2 model loading and token generation
- `kvDiskSim.py` – Simulates disk-based KV cache storage and retrieval
- `memoryMonitor.py` – Tracks local and remote memory usage during inference
- `numa_bind.pyx` & `setup.py` – Cython module for setting memory binding to specific NUMA nodes
- `run_inference.sh` – Shell script to automate running experiments across all memory strategies

### Helper Utilities
- `configurator.py` – Parses and updates runtime arguments for `inference.py`
- `plot.py` – Generates performance plots for latency and throughput

---

## ⚙️ Workload Generation Parameters

You can configure synthetic workloads using:

| Parameter           | Description                                 |
|---------------------|---------------------------------------------|
| `lambda_rate`       | Average number of requests per second       |
| `simulation_duration` | Total simulation duration in seconds      |
| `new_conv_prob`     | Probability of starting a new conversation  |
| `seed`              | Random seed for reproducibility             |

**Example:**
```bash
python workloadGen.py --lambda_rate 5 --simulation_duration 50 --new_conv_prob 0.7 --seed 42
```

---

## 🚀 Inference Configuration Parameters

Set the following arguments when running inference:

| Parameter              | Description                                             | Default |
|------------------------|---------------------------------------------------------|---------|
| `init_from`            | GPT-2 model variant (`gpt2`, `gpt2-medium`, etc.)       | `gpt2` |
| `start`                | Prompt input or prompt file path                        | `FILE:data/input.txt` |
| `input_tokens`         | Max input token length                                  | `500`   |
| `max_new_tokens`       | Max number of tokens to generate                        | `20`    |
| `temperature`          | Controls randomness (<1.0 = deterministic)              | `0.0`   |
| `top_k`                | Top-k sampling (ignored if temperature = 0)             | `200`   |
| `kv_method`            | Memory strategy (`local-memory`, `remote-memory`, `disk`, `tiered-lru`) | `local-memory` |
| `tiered_kv_cache`      | Enable naive tiered cache                               | `False` |
| `lru_tiered_kv_cache`  | Enable LRU-based tiered cache                           | `False` |
| `kv_cache_dir`         | Directory for disk-based KV cache                       | `./kv_cache_disk/` |
| `device`               | Computation device (`cpu` or `cuda`)                    | `cpu`   |
| `dtype`                | Data type (`bfloat16`, `float16`, etc.)                 | Auto-detect |
| `seed`                 | Random seed for reproducibility                         | `42`    |

### Memory Limit Parameters (Tiered Systems)

| Parameter                     | Description                         | Default |
|------------------------------|-------------------------------------|---------|
| `memory_limit`               | Memory limit (MB) for naive tiered  | `1024`  |
| `memory_threshold`           | Usage threshold before spilling     | `0.7`   |
| `lru_local_limit_mb`         | Local memory limit for LRU tiered   | `1024`  |
| `lru_local_threshold`        | Threshold for local memory (LRU)    | `0.7`   |
| `lru_remote_limit_mb`        | Remote memory limit for LRU tiered  | `1024`  |
| `lru_remote_threshold`       | Threshold for remote memory (LRU)   | `0.7`   |
| `local_node` / `remote_node` | NUMA node IDs for memory allocation | `0` / `1` |

**Example:**
```bash
python inference.py --kv_method=remote-memory
```

---

## 🧠 Memory Placement Strategies

This project supports several memory configurations to evaluate how memory placement affects LLM inference performance. You can run each configuration using the examples below:

### 1. Local Memory Only
All KV cache is stored in local memory (NUMA node 0).

```bash
numactl --cpunodebind=0 --membind=0 python inference.py
```

### 2. Remote Memory Only
Simulates disaggregated memory by placing the KV cache on a remote NUMA node (node 1), while computation runs on node 0.

```bash
numactl --cpunodebind=0 python inference.py --kv_method=remote-memory
```

### 3. Disk Only
KV cache is stored and fetched from disk during inference. This simulates running under strict memory constraints.

```bash
numactl --cpunodebind=0 python inference.py --kv_method=disk
```

### 4. Tiered Memory System (Naive)
KV cache is placed sequentially across local memory → remote memory → disk as each tier reaches capacity.

```bash
numactl --cpunodebind=0 python inference.py --tiered_kv_cache=True
```

### 5. Tiered Memory System with LRU
Same as above, but uses Least Recently Used (LRU) policy for eviction and promotion across tiers, keeping frequently accessed cache blocks in faster memory.

```bash
numactl --cpunodebind=0 python inference.py --kv_method=tiered-lru --lru_tiered_kv_cache=True
```

---

## 📈 Reproducing Experiments

To reproduce all experiments and generate results for different memory configurations, simply run the provided shell script:

```bash
chmod +x run_inference.sh
./run_inference.sh
```
