#!/bin/bash

# Install system-level packages
sudo apt update
sudo apt install -y numactl hwloc libnuma-dev

# Install Python dependencies
pip install -r requirements.txt

# Build the Cython module
python setup.py build_ext --inplace

# Prepare output directory
mkdir -p results
