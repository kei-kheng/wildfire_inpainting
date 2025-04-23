import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Metrics to be plotted
METRICS = {
    "MSE": "Mean Squared Error (MSE)",
    "PSNR": "Peak Signal-to-Noise Ratio (PSNR), dB", 
    "SSIM": "Structural Similarity Index Measure (SSIM)"}

# Constants for plotting
CONSTS = {
    "FONT_SIZE": 12,
    "WIDTH": 10,
    "HEIGHT": 5,
    "DPI": 300,
    "POINT_SIZE": 3,
    "LINE_WIDTH": 2.0,
}

# Set up plot style
plt.rcParams.update({
    "font.size": CONSTS["FONT_SIZE"],
    "figure.figsize": (CONSTS["WIDTH"], CONSTS["HEIGHT"]),
    "figure.dpi": CONSTS["DPI"],
    "lines.markersize": CONSTS["POINT_SIZE"] * 2,
    "lines.linewidth": CONSTS["LINE_WIDTH"],
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], # Overleaf font
})

# For each model, for each run
def collect_data(base_dir, models, num_runs):
    all_data = []
    for model in models:
        for i in range(1, num_runs + 1):
            csv_path = os.path.join(base_dir, model, f"run{i}", "training_log.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df["Model"] = model
                df["Run"] = f"run {i}"
                df["Epoch"] = df["Epoch"].astype(int)
                all_data.append(df)
            else:
                print(f"⚠️ File not found: {csv_path}")
    return pd.concat(all_data, ignore_index=True)

def plot_median_metrics(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Group by "Model" and "Epoch". For each metric, compute median along these two axes.
    median_df = df.groupby(["Model", "Epoch"])[list(METRICS.keys())].median().reset_index()

    sns.set_theme(style="whitegrid", font_scale=1.2)

    for metric, label in METRICS.items():
        plt.figure(figsize=(CONSTS["WIDTH"], CONSTS["HEIGHT"]))
        sns.lineplot(data=median_df, x="Epoch", y=metric, hue="Model")
        plt.title(f"{label} vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(label)
        plt.tight_layout()

        for ext in ["pdf", "svg", "png"]:
            filename = f"{metric.lower()}_median_comparison.{ext}"
            path = os.path.join(output_dir, filename)
            plt.savefig(path, dpi=CONSTS["DPI"] if ext == "png" else None)

        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="results")
    parser.add_argument("--num_runs", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="comparison_plots")
    args = parser.parse_args()

    models = ["CAE", "PCAE", "GAN"]
    df = collect_data(args.base_dir, models, args.num_runs)
    plot_median_metrics(df, args.output_dir)
    print("Plotted graphs from CSV")

if __name__ == "__main__":
    main()