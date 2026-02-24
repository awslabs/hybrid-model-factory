#!/bin/bash

# example:
# Master node: bash examples/train_launchers/multinode_launcher.sh config.yaml 2 0 111.22.33.444
# Second node: bash examples/train_launchers/multinode_launcher.sh config.yaml 2 1 111.22.33.445
# Check if a YAML file is provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <path_to_yaml_file> <master_addr> <nnodes> <node_rank> [master_port]"
    echo "Arguments:"
    echo "  path_to_yaml_file  : Path to the training configuration YAML file"
    echo "  master_addr       : IP address of the master node"
    echo "  nnodes            : Total number of nodes"
    echo "  node_rank         : Rank of this node (0 to nnodes-1)"
    echo "  master_port       : Port for communication (optional, default: 29500)"
    exit 1
fi

yaml_file=$1
master_addr=$2
nnodes=$3
node_rank=$4
master_port=${5:-29500}  # Set default to 29500 if not specified

# Extract output_dir from YAML file
output_dir=$(grep "^output_dir:" "$yaml_file" | sed 's/^output_dir:\s*//; s/["'"'"']//g')

# Create a config for each rank.
# We specify a unique cache folder for each rank as using the same one for multiple ranks often leads to issues.
# Save new config in node_configs folder.
mkdir -p "${output_dir}/node_configs"

# Extract the original cache_dir from yaml file
cache_dir=$(grep "^cache_dir:" "$yaml_file" | sed 's/^cache_dir:\s*//; s/["'"'"']//g')

# Remove trailing slash if it exists
cache_dir="${cache_dir%/}"

# Create new yaml file for this rank
new_yaml="${output_dir}/node_configs/rank_${node_rank}.yaml"

# Copy original yaml to new file
cp "$yaml_file" "$new_yaml"

# Replace cache_dir in the new yaml with updated path
# Using different delimiters (|) since path contains forward slashes
sed -i "s|^cache_dir:.*|cache_dir: ${cache_dir}/rank_${node_rank}|" "$new_yaml"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Generate timestamp for unique log file name
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${output_dir}/training_log_${timestamp}_rank${node_rank}.txt"

# Run the training command and save logs
echo "Starting training with logs saved to: $log_file."
echo "Num nodes: $nnodes, Node rank: $node_rank, Master address: $master_addr, Master port: $master_port"
export DEEPSPEED_DISABLE_ALLGATHER_PARTITIONS=True
FORCE_TORCHRUN=1 NNODES=$nnodes NODE_RANK=$node_rank MASTER_ADDR="$master_addr" MASTER_PORT=$master_port hybridfactory train "$new_yaml"
