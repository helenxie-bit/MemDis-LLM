import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
results_dir = "results"
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# CSVs to calculate averages from
avg_filenames = [
    "metrics_False_local-memory.csv",
    "metrics_False_remote-memory.csv",
    "metrics_False_disk.csv"
]

# CSVs to compare line plots
lineplot_files = {
    "Tiered Memory System": "metrics_True_local-memory.csv",
    "Disk-Based Offloading": "metrics_False_disk.csv"
}

# Metrics and labels
metrics = {
    "time_to_first_token (s)": "Time to First Token (s)",
    "time_between_tokens (s)": "Time Between Tokens (s)",
    "tokens_per_second": "Tokens per Second"
}

# --- Bar Plots: Average Metrics Across Configurations ---
avg_data = {
    "Config": [],
    "time_to_first_token (s)": [],
    "time_between_tokens (s)": [],
    "tokens_per_second": []
}

for filename in avg_filenames:
    path = os.path.join(results_dir, filename)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # Clean column names

    config = filename.split("_")[-1].replace(".csv", "")
    avg_data["Config"].append(config)
    for metric in metrics:
        avg_data[metric].append(df[metric].mean())

avg_df = pd.DataFrame(avg_data)

# Plot each metric as a bar chart
for metric, label in metrics.items():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        avg_df["Config"],
        avg_df[metric],
        color="darkgray",
        edgecolor="black",
        width=0.5
    )
    ax.set_xlabel("KV Cache Placement")
    ax.set_ylabel(label)
    ax.set_title(f"Comparison of {label}")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{metric}_bar_plot.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()

# --- Line Plots: Metric Variation Across Requests ---
# Load both datasets for comparison
df_comp = {}
for label, filename in lineplot_files.items():
    path = os.path.join(results_dir, filename)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df_comp[label] = df

# Plot each metric as a line plot
for metric, label_y in metrics.items():
    plt.figure(figsize=(8, 4))
    for label_x, df in df_comp.items():
        plt.plot(
            df[metric],
            label=label_x,
            marker='o' if "Tiered" in label_x else 'x',
            linestyle='-' if "Tiered" in label_x else '--',
            color='#1F497D' if "Tiered" in label_x else 'gray'
        )

    plt.xlabel("Request Index")
    plt.ylabel(label_y)
    plt.title(f"{label_y} Over 30 Requests")
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{metric}_line_plot.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()
