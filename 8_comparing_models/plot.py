import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = "results"
MODELS = ["CAE", "PCAE", "GAN"]
RUNS = [f"run{i}" for i in range(1, 6)]
METRICS = ["MSE", "PSNR", "SSIM"]

# Collect all results
all_data = []
for model in MODELS:
    for run in RUNS:
        csv_path = os.path.join(BASE_DIR, model, run, "training_log.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["Model"] = model
            df["Run"] = run
            df["Epoch"] = df["Epoch"].astype(int)
            for metric in METRICS:
                df[metric] = pd.to_numeric(df[metric], errors="coerce")
            all_data.append(df)
        else:
            print(f"Missing: {csv_path}")

# Combine all data
combined = pd.concat(all_data)

# Average over runs by Model and Epoch
grouped = (
    combined.groupby(["Model", "Epoch"])[METRICS]
    .mean()
    .reset_index()
)

# Plot using seaborn
sns.set_theme(style="whitegrid", font_scale=1.2)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

for i, metric in enumerate(METRICS):
    sns.lineplot(
        data=grouped,
        x="Epoch",
        y=metric,
        hue="Model",
        ax=axes[i]
    )
    axes[i].set_title(f"{metric} vs Epoch")
    axes[i].set_xlabel("Epoch")
    axes[i].set_ylabel(metric)

plt.tight_layout()
plt.show()
