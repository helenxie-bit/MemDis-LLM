#!/bin/bash

# Configurable parameters
MEMORY_LIMIT_MB=1024
MEMORY_THRESHOLD=0.7
NUM_REQUESTS=100
INFER_SCRIPT="inference.py"
THRESHOLD_MB=$(echo "$MEMORY_LIMIT_MB * $MEMORY_THRESHOLD" | bc)
MEMORY_TIER="local"  # Start from local tier
CPU_NODE=0
CURRENT_NODE=0

# Get current memory usage (used = total - free)
get_numa_node_used_mb() {
  NODE_ID=$1
  TOTAL=$(numactl -H | awk -v node="node $NODE_ID size" '$0 ~ node {print $(NF-1)}')
  FREE=$(numactl -H | awk -v node="node $NODE_ID free" '$0 ~ node {print $(NF-1)}')
  echo "$(echo "$TOTAL - $FREE" | bc)"
}

for ((i = 1; i <= NUM_REQUESTS; i++)); do
  echo "[INFO] Request $i of $NUM_REQUESTS | Current tier: $MEMORY_TIER"

  if [ "$MEMORY_TIER" == "local" ]; then
    USED_MB=$(get_numa_node_used_mb 0)
    if (( $(echo "$USED_MB >= $THRESHOLD_MB" | bc -l) )); then
      echo "[SWITCH] Local tier full → switching to remote tier"
      MEMORY_TIER="remote"
      CURRENT_NODE=1
    fi
  fi

  if [ "$MEMORY_TIER" == "remote" ]; then
    USED_MB=$(get_numa_node_used_mb 1)
    if (( $(echo "$USED_MB >= $THRESHOLD_MB" | bc -l) )); then
      echo "[SWITCH] Remote tier full → switching to disk tier"
      MEMORY_TIER="disk"
      CURRENT_NODE=0
    fi
  fi

  # Run the inference based on current memory tier
  if [ "$MEMORY_TIER" == "local" ]; then
    numactl --cpunodebind=$CPU_NODE --membind=$CURRENT_NODE python $INFER_SCRIPT --num_requests=1
  elif [ "$MEMORY_TIER" == "remote" ]; then
    numactl --cpunodebind=$CPU_NODE --membind=$CURRENT_NODE python $INFER_SCRIPT --num_requests=1
  else
    numactl --cpunodebind=$CPU_NODE --membind=$CURRENT_NODE python $INFER_SCRIPT --kv_method=disk --num_requests=1
  fi
done
