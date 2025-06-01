import subprocess

def get_numastat(pid):
    try:
        result = subprocess.run(["numastat", "-p", str(pid)], capture_output=True, text=True, check=True)
        #print(f"NUMA statistics for PID {pid}:\n{result.stdout}")
        lines = result.stdout.splitlines()
        for line in lines:
            if line.strip().startswith("Total"):
                parts = line.split()
                # Assuming format: 'Total', <Node0>, <Node1>, <Total>
                return {
                    "Node0_MB": float(parts[1]),
                    "Node1_MB": float(parts[2]),
                    "Total_MB": float(parts[3])
                }

        print("Could not find 'Total' line in numastat output.")
        return {}
    except Exception as e:
        print(f"Error reading numastat: {e}")
        return {}