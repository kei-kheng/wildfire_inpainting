import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset

# Dataset
class IR_Images(Dataset):
    def __init__(self, data_dir, subfolders, transform=None):
        self.files = []
        self.transform = transform

        for folder in subfolders:
            folder_path = os.path.join(data_dir, folder)
            imgs_in_folder = sorted(glob.glob(os.path.join(folder_path, "*.png")))
            self.files.extend(imgs_in_folder)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        # Random rotation, https://pytorch.org/vision/main/generated/torchvision.transforms.functional.rotate.html
        img = T.functional.rotate(img, random.choice((0, 90, 180, 270)))

        if self.transform:
            img = self.transform(img)
        # Return filename for easier reference when running: https://www.geeksforgeeks.org/python-os-path-basename-method/
        return img, os.path.basename(img_path)

# Resize image, retain aspect ratio with specified longest side
# Ensures final height and width are both divisible by 2^n, where n = upsampling/downsampling steps => Modify multiple if n changes
class ImageResize:
    def __init__(self, scaled_dim, multiple=16):
        self.scaled_dim = scaled_dim
        self.multiple = multiple

    def __call__(self, img):
        w, h = img.size
        new_w = 0
        new_h = 0

        if w >= h:
            new_w = self.scaled_dim
            new_h = int(h * (self.scaled_dim / float(w)))
        else:
            new_h = self.scaled_dim
            new_w = int(w * (self.scaled_dim / float(h)))
        
        snapped_w = max(self.multiple, (new_w // self.multiple) * self.multiple)
        snapped_h = max(self.multiple, (new_h // self.multiple) * self.multiple)

        return T.Resize((snapped_h, snapped_w))(img)  # Apply transform to image

# Getter function that returns transform
def get_transform(scaled_dim=320):
    return T.Compose([
        ImageResize(scaled_dim=scaled_dim),
        T.ToTensor(),
        # Scale output to [-1, 1]
        # Reference: https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html
        # Reference: https://discuss.pytorch.org/t/understanding-transform-normalize/21730/21?page=2
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

# Create 1-4 random square masks (may overlap) with varying sizes that total up to coverage * total_pixels
def create_random_square_masks(channels, height, width, coverage):
    mask_1ch = np.ones((height, width), dtype=np.float32)  # Single-channel mask

    total_pixels = height * width
    num_squares = np.random.randint(1, 5)  # Random number from 1-4

    coverage_shares = np.random.random(num_squares)  # Returns a 'num_squares'-element NumPy array of floats in interval [0.0, 1.0)]
    coverage_shares /= coverage_shares.sum()  # Want random numbers to sum to 1
    coverage_shares *= coverage  # Scale by 'coverage' so the random numbers sum to 'coverage'

    for share in coverage_shares:
        square_area = int(share * total_pixels)
        side_length = max(int(np.sqrt(square_area)), 1)
        half_side = side_length // 2

        # Randomly pick a center  within valid boundaries
        center_x = np.random.randint(half_side, height - half_side)
        center_y = np.random.randint(half_side, width - half_side)
        start_x = center_x - half_side
        end_x = start_x + side_length
        start_y = center_y - half_side
        end_y = start_y + side_length

        # Zero out that square area
        mask_1ch[start_x:end_x, start_y:end_y] = 0.0

    # Replicate for the specified number of channels
    mask = np.repeat(mask_1ch[np.newaxis, :, :], channels, axis=0)
    return mask

# Mask application
def apply_mask(batch, coverage):
    B, _, H, W = batch.shape
    masked_list = []
    mask_list = []

    for i in range(B):
        mask = create_random_square_masks(3, H, W, coverage)
        mask_tensor = torch.from_numpy(mask).to(batch.device)
        masked_img = batch[i] * mask_tensor
        masked_list.append(masked_img.unsqueeze(0))
        mask_list.append(mask_tensor.unsqueeze(0))

    masked = torch.cat(masked_list, dim=0)  # (B,3,H,W)
    masks = torch.cat(mask_list, dim=0)  # (B,3,H,W)
    return masked, masks

def plot_from_csv(output_dir, csv_file="training_log.csv"):
    # Load CSV as Pandas DataFrame
    csv_path = f"results/{output_dir}/{csv_file}"
    df = pd.read_csv(csv_path)
    # Group by epoch, compute average loss per epoch
    df_avg = df.groupby("Epoch")[["LossD", "LossG", "LossG_recon"]].mean()

    # Plot LossD and LossG against epoch
    plt.figure(figsize=(10, 5))
    # Syntax: plot(x-axis values, y-axis values, legend label)
    # 'Epoch' would be the index due to groupby()
    plt.plot(df_avg.index, df_avg["LossD"], label="Discriminator Loss (LossD)")
    plt.plot(df_avg.index, df_avg["LossG"], label="Generator Loss (LossG)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LossD and LossG per Epoch")
    plt.legend()
    plt.grid(True)
    # Save as PNG
    plt.savefig(f"results/{output_dir}/lossD_lossG_vs_epoch.png", dpi=300)
    plt.show()

    # Plot LossG_recon against epoch
    plt.figure(figsize=(10, 5))
    plt.plot(df_avg.index, df_avg["LossG_recon"], label="Reconstruction Loss (LossG_recon)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{output_dir}/lossG_recon_vs_epoch.png", dpi=300)
    plt.show()