import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    "LRU Memory System": "metrics_True_tiered-lru.csv",
    "Disk Baseline": "metrics_False_disk.csv"
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

    config = filename.replace("_0.5", "").split("_")[-1].replace(".csv", "")
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
        # Define color and style based on system type
        if "Disk Baseline" in label_x:
            color = 'red'
            linestyle = ':'
            marker = 's'
        elif "Tiered" in label_x:
            color = '#1F497D'
            linestyle = '-'
            marker = 'o'
        else:  # LRU Memory System
            color = 'gray'
            linestyle = '--'
            marker = 'x'
            
        plt.plot(
            df[metric],
            label=label_x,
            marker=marker,
            linestyle=linestyle,
            color=color
        )

    plt.xlabel("Request Index")
    plt.ylabel(label_y)
    plt.title(f"{label_y} Over 25 Simulated Ticks")
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{metric}_line_plot.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()

# --- Comprehensive Line Plots: All Memory Systems Comparison ---
# Load all datasets for comprehensive comparison
all_lineplot_files = {
    "Local Memory Only": "metrics_False_local-memory.csv",
    "Remote Memory Only": "metrics_False_remote-memory.csv", 
    "Disk Only": "metrics_False_disk.csv",
    "Tiered Memory System": "metrics_True_local-memory.csv",
    "LRU Memory System": "metrics_True_tiered-lru.csv"
}

df_all_comp = {}
for label, filename in all_lineplot_files.items():
    path = os.path.join(results_dir, filename)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df_all_comp[label] = df

# Define distinct colors and markers for each system
system_styles = {
    "Local Memory Only": {'color': '#2E75B6', 'marker': 'o', 'linestyle': '-', 'linewidth': 2},
    "Remote Memory Only": {'color': '#C65911', 'marker': '^', 'linestyle': '-', 'linewidth': 2},
    "Disk Only": {'color': '#A2142F', 'marker': 's', 'linestyle': ':', 'linewidth': 3},
    "Tiered Memory System": {'color': '#77AC30', 'marker': 'D', 'linestyle': '-', 'linewidth': 2.5},
    "LRU Memory System": {'color': '#4DBEEE', 'marker': 'x', 'linestyle': '--', 'linewidth': 2.5}
}

# Plot each metric as comprehensive line plots
for metric, label_y in metrics.items():
    plt.figure(figsize=(12, 6))
    
    for label_x, df in df_all_comp.items():
        style = system_styles[label_x]
        plt.plot(
            df[metric],
            label=label_x,
            marker=style['marker'],
            linestyle=style['linestyle'],
            color=style['color'],
            linewidth=style['linewidth'],
            markersize=6,
            alpha=0.8
        )

    plt.xlabel("Request Index")
    plt.ylabel(label_y)
    plt.title(f"{label_y} Comparison Across All Memory Systems")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{metric}_all_systems_line_plot.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()

# Create smoothed version of comprehensive line plots
window_size_all = 15  # Smaller window for better detail with more lines

for metric, label_y in metrics.items():
    plt.figure(figsize=(12, 6))
    
    for label_x, df in df_all_comp.items():
        # Apply rolling average smoothing
        smoothed_data = df[metric].rolling(window=window_size_all, center=True, min_periods=1).mean()
        style = system_styles[label_x]
        
        plt.plot(
            smoothed_data,
            label=label_x,
            linestyle=style['linestyle'],
            color=style['color'],
            linewidth=style['linewidth'] + 1,
            alpha=0.9
        )

    plt.xlabel("Request Index")
    plt.ylabel(label_y)
    plt.title(f"{label_y} Comparison Across All Memory Systems (Smoothed)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{metric}_all_systems_smoothed_line_plot.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()

# --- Smoothed Line Plots: Enhanced Comparison ---
# Smoothing configuration
window_size = 25  # Adjust this value to control smoothing (higher = more smooth)

# Plot each metric as smoothed line plots for easier comparison
for metric, label_y in metrics.items():
    # Create comparison plot (original vs smoothed)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Original data plot (top panel)
    for label_x, df in df_comp.items():
        # Define color and style based on system type
        if "Disk Baseline" in label_x:
            color = 'red'
            linestyle = ':'
            marker = 's'
        elif "Tiered" in label_x:
            color = '#1F497D'
            linestyle = '-'
            marker = 'o'
        else:  # LRU Memory System
            color = 'gray'
            linestyle = '--'
            marker = 'x'
            
        ax1.plot(
            df[metric],
            label=label_x,
            marker=marker,
            linestyle=linestyle,
            color=color,
            alpha=0.7,
            markersize=3
        )
    
    ax1.set_xlabel("Request Index")
    ax1.set_ylabel(label_y)
    ax1.set_title(f"{label_y} Over 25 Simulated Ticks (Original Data)")
    ax1.legend()
    ax1.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    
    # Smoothed data plot (bottom panel)
    for label_x, df in df_comp.items():
        # Apply rolling average smoothing
        smoothed_data = df[metric].rolling(window=window_size, center=True, min_periods=1).mean()
        
        # Define color and style based on system type
        if "Disk Baseline" in label_x:
            color = 'red'
            linestyle = ':'
            linewidth = 3
        elif "Tiered" in label_x:
            color = '#1F497D'
            linestyle = '-'
            linewidth = 2.5
        else:  # LRU Memory System
            color = 'gray'
            linestyle = '--'
            linewidth = 2.5
        
        ax2.plot(
            smoothed_data,
            label=label_x,
            linestyle=linestyle,
            color=color,
            linewidth=linewidth
        )
    
    ax2.set_xlabel("Request Index")
    ax2.set_ylabel(label_y)
    ax2.set_title(f"{label_y} Over 25 Simulated Ticks (Smoothed - Window Size: {window_size})")
    ax2.legend()
    ax2.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()

    # Save comparison plot
    save_path = os.path.join(output_dir, f"{metric}_line_plot_comparison.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()
    
    # Create standalone smoothed plot for clean presentation
    plt.figure(figsize=(10, 6))
    for label_x, df in df_comp.items():
        smoothed_data = df[metric].rolling(window=window_size, center=True, min_periods=1).mean()
        
        # Define color and style based on system type
        if "Disk Baseline" in label_x:
            color = 'red'
            linestyle = ':'
            linewidth = 4
        elif "Tiered" in label_x:
            color = '#1F497D'
            linestyle = '-'
            linewidth = 3
        else:  # LRU Memory System
            color = 'gray'
            linestyle = '--'
            linewidth = 3
        
        plt.plot(
            smoothed_data,
            label=label_x,
            linestyle=linestyle,
            color=color,
            linewidth=linewidth
        )

    plt.xlabel("Request Index")
    plt.ylabel(label_y)
    plt.title(f"{label_y} Over 25 Simulated Ticks (Smoothed)")
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Save standalone smoothed plot
    save_path = os.path.join(output_dir, f"{metric}_smoothed_line_plot.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()

# --- Comprehensive Summary Comparison ---
# Collect data from all systems for comprehensive comparison
all_systems_data = {
    "System": [],
    "time_to_first_token (s)": [],
    "time_between_tokens (s)": [],
    "tokens_per_second": []
}

# Add the basic configurations (from avg_filenames)
system_names = {
    "metrics_False_local-memory.csv": "Local Memory Only",
    "metrics_False_remote-memory.csv": "Remote Memory Only", 
    "metrics_False_disk.csv": "Disk Only (Baseline)"
}

for filename in avg_filenames:
    path = os.path.join(results_dir, filename)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    
    system_name = system_names[filename]
    all_systems_data["System"].append(system_name)
    for metric in metrics:
        all_systems_data[metric].append(df[metric].mean())

# Add the advanced systems (from lineplot_files)
advanced_systems = {
    "metrics_True_local-memory.csv": "Tiered Memory System",
    "metrics_True_tiered-lru.csv": "LRU Memory System"
}

for filename, display_name in advanced_systems.items():
    path = os.path.join(results_dir, filename)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    
    all_systems_data["System"].append(display_name)
    for metric in metrics:
        all_systems_data[metric].append(df[metric].mean())

# Create summary DataFrame
summary_df = pd.DataFrame(all_systems_data)

# Create comprehensive bar chart comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

colors = ['#2E75B6', '#C65911', '#A2142F', '#77AC30', '#4DBEEE']

for i, (metric, label_y) in enumerate(metrics.items()):
    ax = axes[i]
    bars = ax.bar(
        range(len(summary_df)), 
        summary_df[metric],
        color=colors[:len(summary_df)],
        edgecolor='black',
        width=0.7,
        alpha=0.8
    )
    
    # Add value labels on bars
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel("System Type")
    ax.set_ylabel(label_y)
    ax.set_title(f"Average {label_y}\nAcross All Systems")
    ax.set_xticks(range(len(summary_df)))
    ax.set_xticklabels(summary_df["System"], rotation=45, ha='right')
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

plt.tight_layout()
save_path = os.path.join(output_dir, "comprehensive_summary_bar_chart.png")
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.show()
plt.close()

# Create normalized comparison (relative to disk baseline)
disk_baseline_idx = summary_df[summary_df["System"] == "Disk Only (Baseline)"].index[0]
normalized_data = summary_df.copy()

for metric in metrics:
    baseline_value = summary_df.loc[disk_baseline_idx, metric]
    normalized_data[f"{metric}_normalized"] = summary_df[metric] / baseline_value

# Plot normalized comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (metric, label_y) in enumerate(metrics.items()):
    ax = axes[i]
    norm_metric = f"{metric}_normalized"
    
    bars = ax.bar(
        range(len(normalized_data)), 
        normalized_data[norm_metric],
        color=colors[:len(normalized_data)],
        edgecolor='black',
        width=0.7,
        alpha=0.8
    )
    
    # Add horizontal line at y=1 (baseline)
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Disk Baseline')
    
    # Add value labels on bars
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel("System Type")
    ax.set_ylabel(f"Relative {label_y}\n(vs Disk Baseline)")
    ax.set_title(f"Relative Performance: {label_y}\n(Disk Baseline = 1.0)")
    ax.set_xticks(range(len(normalized_data)))
    ax.set_xticklabels(normalized_data["System"], rotation=45, ha='right')
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend()

plt.tight_layout()
save_path = os.path.join(output_dir, "normalized_performance_comparison.png")
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.show()
plt.close()