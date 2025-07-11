import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Metrics to be plotted
METRICS = {"MSE": "MSE", "PSNR": "PSNR (dB)", "SSIM": "SSIM"}

# Constants for plotting
CONSTS = {
    "FONT_SIZE": 20,
    "TITLE_SIZE": 24,
    "LABEL_SIZE": 24,
    "TICK_SIZE": 20,
    "LEGEND_SIZE": 24,
    "FONT_SCALE": 2.0,
    "WIDTH": 10,
    "HEIGHT": 8,
    "POINT_SIZE": 3,
    "LINE_WIDTH": 2.0,
    "DPI": 300,
}

# Set up plot style
# Set up Seaborn first to not override matplotlib's font settings
sns.set_theme(style="whitegrid", font_scale=CONSTS["FONT_SCALE"])
plt.rcParams.update(
    {
        "font.size": CONSTS["FONT_SIZE"],
        "figure.figsize": (CONSTS["WIDTH"], CONSTS["HEIGHT"]),
        "figure.dpi": CONSTS["DPI"],
        "lines.markersize": CONSTS["POINT_SIZE"] * 2,
        "lines.linewidth": CONSTS["LINE_WIDTH"],
        "axes.titlesize": CONSTS["TITLE_SIZE"],
        "axes.labelsize": CONSTS["LABEL_SIZE"],
        "xtick.labelsize": CONSTS["TICK_SIZE"],
        "ytick.labelsize": CONSTS["TICK_SIZE"],
        "legend.fontsize": CONSTS["LEGEND_SIZE"],
        "legend.title_fontsize": CONSTS["LEGEND_SIZE"],
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],  # Overleaf font
    }
)


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
    median_df = (
        df.groupby(["Model", "Epoch"])[list(METRICS.keys())].median().reset_index()
    )

    for metric, label in METRICS.items():
        plt.figure(figsize=(CONSTS["WIDTH"], CONSTS["HEIGHT"]))

        for model in df["Model"].unique():
            model_df = df[df["Model"] == model]
            pivot = model_df.groupby(["Epoch", "Run"])[metric].mean().unstack()
            median = pivot.median(axis=1)
            min_ = pivot.min(axis=1)
            max_ = pivot.max(axis=1)

            plt.plot(median.index, median.values, label=model)
            plt.fill_between(median.index, min_, max_, alpha=0.2)

        # plt.title(f"{label} vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(label)
        plt.legend(title="Model")
        plt.tight_layout()

        for ext in ["pdf", "svg", "png"]:
            filename = f"{metric.lower()}_median_comparison.{ext}"
            path = os.path.join(output_dir, filename)
            plt.savefig(path, dpi=CONSTS["DPI"] if ext == "png" else None)

        plt.close()


def calculate_variance(df, output_dir, metrics=["MSE", "PSNR", "SSIM"]):
    variance_summary = {}
    for model in df["Model"].unique():
        model_df = df[df["Model"] == model]
        grouped = model_df.groupby(["Run", "Epoch"])[metrics].mean().reset_index()

        for metric in metrics:
            pivot = grouped.pivot(index="Epoch", columns="Run", values=metric)
            if pivot.shape[1] < 2:
                continue  # Skip if less than 2 runs
            epoch_wise_var = pivot.var(axis=1)
            variance_summary.setdefault(model, {})[metric] = {
                "min": float(epoch_wise_var.min()),
                "max": float(epoch_wise_var.max()),
                "mean": float(epoch_wise_var.mean()),
            }

    # Write to file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "variance_summary.txt")
    with open(output_path, "w") as f:
        for model in variance_summary:
            f.write(f"{model}:\n")
            for metric, stats in variance_summary[model].items():
                f.write(
                    f"  {metric} variance -> Min: {stats['min']:.4f}, Max: {stats['max']:.4f}, Mean: {stats['mean']:.4f}\n"
                )
            f.write("\n")
    print(f"Variance summary written to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="results")
    parser.add_argument("--num_runs", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="comparison_plots")
    args = parser.parse_args()

    models = ["CAE", "PCAE", "GAN"]
    df = collect_data(args.base_dir, models, args.num_runs)
    plot_median_metrics(df, args.output_dir)
    # calculate_variance(df, args.output_dir)
    print("Plotted graphs from CSV")


if __name__ == "__main__":
    main()
