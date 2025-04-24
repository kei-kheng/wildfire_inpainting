import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

EPOCH_DIRS = ["150_epochs", "200_epochs"]
RUNS = [f"run{i}" for i in range(1, 6)]
METRICS = {
    "MSE": "Mean Squared Error (MSE)",
    "PSNR": "Peak Signal-to-Noise Ratio (PSNR), dB",
    "SSIM": "Structural Similarity Index Measure (SSIM)",
}
OUT_DIR = "inference_results/150vs200_plots"

STYLE = {
    "FONT_SIZE": 12,
    "FONT_SCALE": 2.0,
    "WIDTH": 14,
    "HEIGHT": 7,
    "DPI": 300,
}

plt.rcParams.update({
    "font.size": STYLE["FONT_SIZE"],
    "figure.figsize": (STYLE["WIDTH"], STYLE["HEIGHT"]),
    "figure.dpi": STYLE["DPI"],
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
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
            print(f"⚠️ Not found: {csv_path}")
    return pd.concat(all_dfs, ignore_index=True)

def prepare_grouped_data(df, metric):
    summary = []
    for epoch_dir in EPOCH_DIRS:
        epoch_df = df[df["EpochDir"] == epoch_dir]
        grouped = epoch_df.groupby(["Folder", "Run"])[metric].mean().reset_index()
        pivot = grouped.pivot(index="Folder", columns="Run", values=metric)

        med = pivot.median(axis=1)
        min_ = pivot.min(axis=1)
        max_ = pivot.max(axis=1)

        summary.append(pd.DataFrame({
            "Folder": med.index.astype(str),
            "EpochSetting": epoch_dir,
            "Median": med.values,
            "Min": min_.values,
            "Max": max_.values,
        }))
    return pd.concat(summary, ignore_index=True)

def plot_grouped_bar(summary_df, metric, ylabel):
    os.makedirs(OUT_DIR, exist_ok=True)
    summary_df["Folder"] = summary_df["Folder"].astype(int)
    summary_df = summary_df.sort_values("Folder")

    # Bar locations
    folders = summary_df["Folder"].unique()
    x = np.arange(len(folders))
    width = 0.35

    fig, ax = plt.subplots()

    for i, setting in enumerate(EPOCH_DIRS):
        data = summary_df[summary_df["EpochSetting"] == setting]
        offsets = x - width / 2 if i == 0 else x + width / 2

        ax.bar(offsets, data["Median"], width=width, label=setting,
               yerr=[data["Median"] - data["Min"], data["Max"] - data["Median"]],
               capsize=5)

    ax.set_xlabel("Folder")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by Folder (Median with Min-Max Range)")
    ax.set_xticks(x)
    ax.set_xticklabels(folders, rotation=90)
    ax.legend(title="Epochs")
    fig.tight_layout()

    for ext in ["png", "svg", "pdf"]:
        plt.savefig(os.path.join(OUT_DIR, f"{metric}_comparison.{ext}"), dpi=STYLE["DPI"] if ext == "png" else None)
    plt.close()

def main():
    df_all = pd.concat([load_all_runs(epoch) for epoch in EPOCH_DIRS], ignore_index=True)

    for metric, ylabel in METRICS.items():
        summary = prepare_grouped_data(df_all, metric)
        plot_grouped_bar(summary, metric, ylabel)

    print("Inference comparison plots saved to:", OUT_DIR)

if __name__ == "__main__":
    main()