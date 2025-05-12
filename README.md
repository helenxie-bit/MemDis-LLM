# MemDis-LLM
MemDis-LLM explores the integration of memory disaggregation into large language model (LLM) inference by implementing a tiered memory system. This system prioritizes the use of disaggregated memory over disk-based offloading when local memory is insufficient.

## Inference Scenarios
We evaluate LLM inference under three memory configurations in order to investigate the effectiveness of memory disaggregation as an alternative to disk-based offloading.

### Case 1 (Baseline): Local Memory
The model runs entirely within local memory. This serves as the baseline for comparison.

### Case 2: Remote Memory
We simulate disaggregated memory by fetching KV cache from a remote NUMA node while computation remains on the local node. This models the latency and bandwidth characteristics of disaggregated memory systems.

### Case 3: Disk-based Offloading
KV cache is written to and read from disk during inference. For each token and each layer, the KV cache is fetched from disk, mimicking extreme memory constraints.

## How to Run the Experiment
```[shell]
chmod +x run_inference.sh
./run_inference.sh
```
