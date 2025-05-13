# Install packages needed for NUMA node
sudo apt update
sudo apt install -y numactl hwloc

# Install dependencies
pip install -r requirements.txt

# Create a directory for the results
mkdir -p results
