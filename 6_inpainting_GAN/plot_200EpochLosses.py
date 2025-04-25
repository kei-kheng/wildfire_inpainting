import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = "results/200_epochs"
OUTPUT_DIR = "results/plot_200EpochLosses"
RUNS = [f"run{i}" for i in range(1, 6)]
METRICS = {
    "LossD": "Discriminator Loss (LossD)",
    "LossG": "Generator Loss (LossG)",
    "LossG_recon": "Reconstruction Loss (LossG_recon)",
    "MSE": "Mean Squared Error (MSE)",
    "PSNR": "Peak Signal-to-Noise Ratio (PSNR), dB",
    "SSIM": "Structural Similarity Index (SSIM)",
}

STYLE = {
    "FONT_SIZE": 12,
    "FONT_SCALE": 2.0,
    "WIDTH": 14,
    "HEIGHT": 7,
    "DPI": 300,
    "POINT_SIZE": 3,
    "LINE_WIDTH": 2.0,
}

sns.set_theme(style="whitegrid", font_scale=STYLE["FONT_SCALE"])
plt.rcParams.update({
    "font.size": STYLE["FONT_SIZE"],
    "figure.figsize": (STYLE["WIDTH"], STYLE["HEIGHT"]),
    "figure.dpi": STYLE["DPI"],
    "lines.markersize": STYLE["POINT_SIZE"] * 2,
    "lines.linewidth": STYLE["LINE_WIDTH"],
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

def load_data():
    all_dfs = []
    for run in RUNS:
        path = os.path.join(BASE_DIR, run, "training_log.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["Run"] = run
            all_dfs.append(df)
        else:
            print(f"⚠️ File not found: {path}")
    return pd.concat(all_dfs, ignore_index=True)

def plot_losses_together(df):
    grouped = df.groupby(["Epoch", "Run"])[["LossD", "LossG"]].mean().reset_index()
    pivot_D = grouped.pivot(index="Epoch", columns="Run", values="LossD")
    pivot_G = grouped.pivot(index="Epoch", columns="Run", values="LossG")

    med_D, min_D, max_D = pivot_D.median(axis=1), pivot_D.min(axis=1), pivot_D.max(axis=1)
    med_G, min_G, max_G = pivot_G.median(axis=1), pivot_G.min(axis=1), pivot_G.max(axis=1)

    plt.figure()
    plt.plot(med_D.index, med_D.values, label=r"$\mathcal{L}_{D}$ (Median)", color="blue")
    plt.fill_between(med_D.index, min_D, max_D, alpha=0.2, color="blue")

    plt.plot(med_G.index, med_G.values, label=r"$\mathcal{L}_{G}$ (Median)", color="red")
    plt.fill_between(med_G.index, min_G, max_G, alpha=0.2, color="red")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(r"Discriminator($\mathcal{L}_{D}$) and Generator Loss ($\mathcal{L}_{G}$) vs Epoch")
    plt.legend()
    plt.tight_layout()

    for ext in ["png", "svg", "pdf"]:
        plt.savefig(os.path.join(OUTPUT_DIR, f"LossD_LossG_vs_epoch.{ext}"), dpi=STYLE["DPI"] if ext == "png" else None)
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()

    plot_losses_together(df)
    print("Plots saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
