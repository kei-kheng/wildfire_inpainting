import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Directory
EPOCH_DIRS = ["150_epochs", "200_epochs"]

RUNS = [f"run{i}" for i in range(1, 6)]

# Metrics and y-axis labels
METRICS = {
    "MSE": "MSE",
    "PSNR": "PNSR (dB)",
    "SSIM": "SSIM",
}

#Output directory
OUT_DIR = "inference_results/150vs200_plots"

# Set up plot style
STYLE = {
    "FONT_SIZE": 12, 
    "TITLE_SIZE": 26,   
    "LABEL_SIZE": 20,    
    "TICK_SIZE": 12,     
    "LEGEND_SIZE": 14,       
    "FONT_SCALE": 2.0,
    "WIDTH": 14,
    "HEIGHT": 21,
    "POINT_SIZE": 3,
    "LINE_WIDTH": 2.0,
    "DPI": 300,
}

# Set up Seaborn before matplotlib to not override font
sns.set_theme(style="whitegrid", font_scale=STYLE["FONT_SCALE"])

plt.rcParams.update({
    "font.size": STYLE["FONT_SIZE"],
    "figure.figsize": (STYLE["WIDTH"], STYLE["HEIGHT"]),
    "figure.dpi": STYLE["DPI"],
    "lines.markersize": STYLE["POINT_SIZE"] * 2,
    "lines.linewidth": STYLE["LINE_WIDTH"],
    "axes.titlesize": STYLE["TITLE_SIZE"],
    "axes.labelsize": STYLE["LABEL_SIZE"],
    "xtick.labelsize": STYLE["TICK_SIZE"],
    "ytick.labelsize": STYLE["TICK_SIZE"],
    "legend.fontsize": STYLE["LEGEND_SIZE"],
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], # Overleaf font
})

def load_all_runs(base_dir):
    all_dfs = []
    for run in RUNS:
        csv_path = os.path.join("inference_results", base_dir, run, "inference_log.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["Run"] = run
            df["EpochDir"] = base_dir
            all_dfs.append(df)
        else:
            print(f"File not found: {csv_path}")
    return pd.concat(all_dfs, ignore_index=True)

def plot_combined_boxplots(df):
    os.makedirs(OUT_DIR, exist_ok=True)
    metrics = list(METRICS.keys())
    labels = list(METRICS.values())

    fig, axes = plt.subplots(3, 1, figsize=(STYLE["WIDTH"], STYLE["HEIGHT"]), sharex=True)

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[i]
        sns.boxplot(data=df, x="Folder", y=metric, hue="Epoch", ax=ax)

        for ax in axes:
            ax.xaxis.grid(True)
            ax.set_axisbelow(True)

        ax.set_ylabel(METRICS[metric])

    # Remove legends from all subplots
    for ax in axes:
        ax.get_legend().remove()

    # Add one shared legend to the first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, title="Epochs", loc="upper right")

    plt.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(os.path.join(OUT_DIR, f"combined_boxplots.{ext}"))
    plt.close()


def main():
    df_all = pd.concat([load_all_runs(epoch) for epoch in EPOCH_DIRS], ignore_index=True)
    df_all["Epoch"] = df_all["EpochDir"].str.replace("_", " ")

    for metric, ylabel in METRICS.items():
        plot_combined_boxplots(df_all)

    print("Inference comparison plots saved to:", OUT_DIR)

if __name__ == "__main__":
    main()