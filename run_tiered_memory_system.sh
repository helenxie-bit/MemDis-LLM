#!/bin/bash

# -------------------------------
# Configurable parameters
# -------------------------------
MEMORY_LIMIT_MB=1024
MEMORY_THRESHOLD=0.7
NUM_REQUESTS=100
INFER_SCRIPT="inference.py"
THRESHOLD_MB=$(echo "$MEMORY_LIMIT_MB * $MEMORY_THRESHOLD" | bc)
MEMORY_TIER="local"  # Start from local tier
CPU_NODE=0
CURRENT_NODE=0

# -------------------------------
# Helper to get free memory of a NUMA node
# -------------------------------
get_numa_node_free_mb() {
  NODE_ID=$1
  numactl -H | awk -v node="node $NODE_ID free" '$0 ~ node {print $(NF-1)}'
}

# -------------------------------
# Record initial free memory for both nodes
# -------------------------------
INITIAL_FREE_NODE0=$(get_numa_node_free_mb 0)
INITIAL_FREE_NODE1=$(get_numa_node_free_mb 1)

# -------------------------------
# Main inference loop
# -------------------------------
for ((i = 1; i <= NUM_REQUESTS; i++)); do
  echo "[INFO] Request $i of $NUM_REQUESTS | Current tier: $MEMORY_TIER"

  if [ "$MEMORY_TIER" == "local" ]; then
    CURRENT_FREE=$(get_numa_node_free_mb 0)
    USED_DELTA=$(echo "$INITIAL_FREE_NODE0 - $CURRENT_FREE" | bc)
    if (( $(echo "$USED_DELTA >= $THRESHOLD_MB" | bc -l) )); then
      echo "[SWITCH] Local tier full → switching to remote tier"
      MEMORY_TIER="remote"
      CURRENT_NODE=1
    fi
  fi

  if [ "$MEMORY_TIER" == "remote" ]; then
    CURRENT_FREE=$(get_numa_node_free_mb 1)
    USED_DELTA=$(echo "$INITIAL_FREE_NODE1 - $CURRENT_FREE" | bc)
    if (( $(echo "$USED_DELTA >= $THRESHOLD_MB" | bc -l) )); then
      echo "[SWITCH] Remote tier full → switching to disk tier"
      MEMORY_TIER="disk"
      CURRENT_NODE=0  # Default for disk; no memory bind
    fi
  fi

  # -------------------------------
  # Run inference based on current tier
  # -------------------------------
  if [ "$MEMORY_TIER" == "local" ] || [ "$MEMORY_TIER" == "remote" ]; then
    numactl --cpunodebind=$CPU_NODE --membind=$CURRENT_NODE python $INFER_SCRIPT --num_requests=1
  else
    numactl --cpunodebind=$CPU_NODE python $INFER_SCRIPT --kv_method=disk --num_requests=1
  fi
done
