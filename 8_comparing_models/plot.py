import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

METRICS = {"PSNR": "Peak Signal-to-Noise Ratio (PSNR), dB", "SSIM": "Structural Similarity Index (SSIM)"}

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
    "font.serif": ["Computer Modern Roman"],
})

def collect_data(base_dir, models, num_runs):
    all_data = []
    for model in models:
        for i in range(1, num_runs + 1):
            csv_path = os.path.join(base_dir, model, f"run {i}", "training_log.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df["Model"] = model
                df["Run"] = f"run {i}"
                df["Epoch"] = df["Epoch"].astype(int)
                all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def plot_median_metrics(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    median_df = df.groupby(["Model", "Epoch"])[list(METRICS.keys())].median().reset_index()

    sns.set_style(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    for i, (metric, label) in enumerate(METRICS.items()):
        sns.lineplot(data=median_df, x="Epoch", y=metric, hue="Model", ax=axes[i])
        axes[i].set_title(f"{label} vs Epoch")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(label)

    plt.tight_layout()
    for ext in ["pdf", "svg", "png"]:
        plt.savefig(os.path.join(output_dir, f"median_comparison.{ext}"), dpi=CONSTS["DPI"] if ext == "png" else None)
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

if __name__ == "__main__":
    main()