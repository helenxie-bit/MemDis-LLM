# MemDis-LLM
MemDis-LLM explores the integration of memory disaggregation into large language model (LLM) inference by implementing a tiered memory system. This system prioritizes the use of disaggregated memory over disk-based offloading when local memory is insufficient.

## Setup
Install required dependencies:
`pip install -r requirements.txt`

## Inference Scenarios
We evaluate LLM inference under three memory configurations in order to investigate the effectiveness of memory disaggregation as an alternative to disk-based offloading.

### Case 1 (Baseline): Local Memory Only
This is the default setting where the model runs entirely in local memory:
```[shell]
python inference.py \
    --init_from=gpt2 \
    --device=cpu \
    --start="What is the answer to life, the universe, and everything?" \
    --num_requests=5 --max_new_tokens=100
```

**Note**: On NUMA-enabled machines, use `numactl` to ensure both CPU and memory are bound to the same local NUMA node (e.g., node 0):
```[shell]
numactl --cpunodebind=0 --membind=0 python inference.py \
    --init_from=gpt2 \
    --device=cpu \
    --start="What is the answer to life, the universe, and everything?" \
    --num_requests=5 --max_new_tokens=100
```

### Case 2: Remote Memory
To simulate disaggregated memory, we use memory from a remote NUMA node while keeping compute on a local one. This mimics the effect of memory disaggregation:
```[shell]
numactl --cpunodebind=0 --membind=1 python inference.py \
    --init_from=gpt2 \
    --device=cpu \
    --start="What is the answer to life, the universe, and everything?" \
    --num_requests=5 --max_new_tokens=100
```

### Case 3: Disk-based Offloading
We are currently working on this.
