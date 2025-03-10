import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("training_log.csv")

# Compute the average loss per epoch
df_avg = df.groupby("Epoch")[["LossD", "LossG", "LossG_recon"]].mean()

# Plot LossD and LossG
plt.figure(figsize=(10, 5))
plt.plot(df_avg.index, df_avg["LossD"], label="Discriminator Loss (LossD)")
plt.plot(df_avg.index, df_avg["LossG"], label="Generator Loss (LossG)")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LossD and LossG per Epoch")
plt.legend()
plt.grid(True)

# Save the first graph as PNG
plt.savefig("lossD_lossG_vs_epoch.png", dpi=300)  # High-quality image
plt.show()

# Plot LossG_recon separately
plt.figure(figsize=(10, 5))
plt.plot(df_avg.index, df_avg["LossG_recon"], label="Reconstruction Loss (LossG_recon)")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Reconstruction Loss per Epoch")
plt.legend()
plt.grid(True)

# Save the second graph as PNG
plt.savefig("lossG_recon_vs_epoch.png", dpi=300)
plt.show()
