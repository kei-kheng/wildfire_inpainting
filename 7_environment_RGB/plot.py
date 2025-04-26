import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ALL_PARAMETERS = {
    # "confidence_decay": ["0_01", "0_004", "0_002"],
    # "confidence_reception": ["0_4", "0_6", "0_8"],
    # "confidence_threshold": ["0_1", "0_2", "0_3"],
    # "no_of_agents": ["1", "20", "40"],
    "noise": ["none", "gaussian", "salt_and_pepper"]
}

PARAMETER_LABELS = {
    "confidence_decay": r"$\alpha$",
    "confidence_reception": r"$\theta_{\mathrm{init}}$",
    "confidence_threshold": r"$\theta_{\mathrm{min}}$",
    "no_of_agents": r"Number of agents",
    "noise": r"Noise"
}

METRICS = {
    "MSE": "MSE",
    "PSNR": "PSNR (dB)",
    "SSIM": "SSIM",
    "Percentage Explored": "Percentage Explored (\%)"
}

RUNS = [f"run{i}" for i in range(1, 21)]

STYLE = {
    "FONT_SIZE": 16,
    "TITLE_SIZE": 20,
    "LABEL_SIZE": 20,
    "TICK_SIZE": 16,
    "LEGEND_SIZE": 20,
    "FONT_SCALE": 2.0,
    "WIDTH": 10,
    "HEIGHT": 8,
    "POINT_SIZE": 3,
    "LINE_WIDTH": 2.0,
    "DPI": 300,
}

sns.set_theme(style="whitegrid", font_scale=STYLE["FONT_SCALE"])

plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
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
    "font.serif": ["Computer Modern Roman"],
})

# Want to replace '_' with '.' for numbers
def prettify_label(value, parameter):
    if parameter == "noise":
        return {
            "none": "None",
            "gaussian": "Gaussian",
            "salt_and_pepper": "Salt-and-pepper"
        }.get(value, value)
    else:
        return value.replace("_", ".")

def load_runs(parameter, value):
    dfs = []
    base_path = f"results/{parameter}/{value}"
    for run in RUNS:
        path = os.path.join(base_path, run, "evaluation_log.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["Run"] = run
            dfs.append(df)
        else:
            print(f"Missing: {path}")
    return dfs

def plot_metric(parameter, values, metric_key, label):
    plt.figure()
    for value in values:
        dfs = load_runs(parameter, value)
        if not dfs:
            continue
        combined = pd.concat(dfs)
        pivot = combined.groupby(["Step", "Run"])[metric_key].mean().unstack()
        line = pivot.median(axis=1)
        min_ = pivot.min(axis=1)
        max_ = pivot.max(axis=1)
        plt.plot(line.index, line.values, label=prettify_label(value, parameter))
        plt.fill_between(line.index, min_, max_, alpha=0.2)

    plt.xlabel("Step")
    plt.ylabel(label)
    # plt.title(f"{label} over Step")
    plt.legend(title=PARAMETER_LABELS.get(parameter, parameter))
    plt.tight_layout()
    out_dir = f"results/plots/{parameter}"
    os.makedirs(out_dir, exist_ok=True)
    for ext in ["pdf", "png", "svg"]:
        plt.savefig(os.path.join(out_dir, f"{metric_key}_vs_step.{ext}"))
    plt.close()

def main():
    for parameter, values in ALL_PARAMETERS.items():
        for metric_key, label in METRICS.items():
            plot_metric(parameter, values, metric_key, label)
    print("All parameter sweep plots saved.")

if __name__ == "__main__":
    main()
