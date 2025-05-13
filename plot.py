import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing your CSV files
results_dir = "results"
filenames = [
    "metrics_False_local-memory.csv",
    "metrics_False_remote-memory.csv",
    "metrics_False_disk.csv"
]
os.makedirs("plots", exist_ok=True)

# Initialize dictionary to hold averages
averages = {
    "Config": [],
    "time_to_first_token (s)": [],  # Time to First Token (s)
    "time_between_tokens (s)": [],   # Time Between Tokens (s)
    "tokens_per_second": []    # Tokens per Second
}

# Read and compute means
for fname in filenames:
    path = os.path.join(results_dir, fname)
    df = pd.read_csv(path)

    # Ensure clean column names
    df.columns = df.columns.str.strip()

    config = fname.split("_")[-1].replace(".csv", "")
    averages["Config"].append(config)
    averages["time_to_first_token (s)"].append(df["time_to_first_token (s)"].mean())
    averages["time_between_tokens (s)"].append(df["time_between_tokens (s)"].mean())
    averages["tokens_per_second"].append(df["tokens_per_second"].mean())

# Convert to DataFrame for plotting
avg_df = pd.DataFrame(averages)

# Plotting settings
metrics = {
    "time_to_first_token (s)": "Time to First Token (s)",
    "time_between_tokens (s)": "Time Between Tokens (s)",
    "tokens_per_second": "Tokens per Second"
}
colors = ['darkgray', 'darkgray', 'darkgray']

for i, (key, label) in enumerate(metrics.items()):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        avg_df["Config"],
        avg_df[key],
        color=colors[i],
        edgecolor="black",
        width=0.5
    )
    ax.set_ylabel(label)
    ax.set_title(f"Comparison of {label}")
    ax.set_xlabel("KV Cache Placement")
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join("plots", f"{key}_comparison_plot.png")
    #plt.savefig(save_path, bbox_inches="tight")

    #plt.show()

# File paths
file_1 = "results/metrics_True_local-memory.csv"
file_2 = "results/metrics_False_disk.csv"

# Read CSVs and clean column names
df1 = pd.read_csv(file_1)
df2 = pd.read_csv(file_2)
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# Define labels
label1 = "Tiered Memory System"
label2 = "Disk-Based Offloading"

# Plot each metric
for key, label in metrics.items():
    plt.figure(figsize=(8, 4))
    plt.plot(df1[key], label=label1, marker='o', linestyle='-', color='#1F497D')
    plt.plot(df2[key], label=label2, marker='x', linestyle='--', color='gray')

    plt.title(f"Comparison of {label} Over 30 Requests")
    plt.xlabel("Request Index")
    plt.ylabel(label)
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Save plot
    save_path = os.path.join("plots", f"{key}_line_comparison.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()