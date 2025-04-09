import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset
from skimage.metrics import structural_similarity as SSIM

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
    num_squares = np.random.randint(1, 5)  # 1-4 square masks

    coverage_shares = np.random.random(num_squares)  # Returns a 'num_squares'-element NumPy array of floats in interval [0.0, 1.0)]
    coverage_shares /= coverage_shares.sum()  # Want random numbers to sum to 1
    coverage_shares *= coverage  # Scale by 'coverage' so the random numbers sum to 'coverage'

    for share in coverage_shares:
        square_area = int(share * total_pixels)
        side_length = max(int(np.sqrt(square_area)), 1)
        # Ensure each square is at least 8x8 for SSIM calculation
        side_length = max(side_length, 8)
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

def cal_MSE(composite_image, ground_truth):
    MSE = np.mean((composite_image - ground_truth)**2)
    if MSE < 1e-10:
        return 100.0
    return MSE

def cal_PSNR(composite_image, ground_truth):
    data_range = ground_truth.max() - ground_truth.min()
    MSE = cal_MSE(composite_image, ground_truth)
    PSNR = 10 * np.log10((data_range)**2 / MSE)
    return PSNR

def cal_SSIM(composite_image, ground_truth, win_size=7):
    data_range = ground_truth.max() - ground_truth.min()
    SSIM_value = SSIM(ground_truth, composite_image, data_range=data_range, channel_axis=0, win_size=win_size)
    return SSIM_value

def plot_from_csv_training(output_dir, csv_file="training_log.csv"):
    # Load CSV as Pandas DataFrame
    csv_path = f"results/{output_dir}/{csv_file}"
    df = pd.read_csv(csv_path)
    # Group by epoch, compute average loss per epoch
    df_avg = df.groupby("Epoch")[["LossD", "LossG", "LossG_recon", "MSE", "PSNR", "SSIM"]].mean().reset_index()
    os.makedirs(f"results/{output_dir}/plots", exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.2)

    metrics = {
        "LossG_recon": "Reconstruction Loss (LossG_recon)",
        "MSE": "Mean Squared Error (MSE)",
        "PSNR": "Peak Signal-to-Noise Ratio (PSNR), dB",
        "SSIM": "Structural Similarity Index (SSIM)",
    }

    # Combine LossD and LossG into one long-form DataFrame for Seaborn
    loss_df = df_avg[["Epoch", "LossD", "LossG"]].melt(id_vars="Epoch", var_name="Loss Type", value_name="Loss")
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=loss_df, x="Epoch", y="Loss", hue="Loss Type", linewidth=2.0)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Adversarial Losses: Discriminator Loss (LossD) and Generator Loss (LossG) vs Epoch")
    plt.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(f"results/{output_dir}/plots/LossD_LossG_vs_epoch.{ext}", dpi=300 if ext == "png" else None)
    plt.close()

    for key, label in metrics.items():
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df_avg, x="Epoch", y=key, linewidth=2.0)
        plt.xlabel("Epoch")
        plt.ylabel(label)
        plt.title(f"{label} vs Epoch")
        plt.tight_layout()
        for ext in ["png", "svg", "pdf"]:
            plt.savefig(f"results/{output_dir}/plots/{key}_vs_epoch.{ext}", dpi=300 if ext == "png" else None)
        plt.close()

def plot_from_csv_inferencing(output_dir, csv_file="inference_log.csv"):
    csv_path = f"inference_results/{output_dir}/{csv_file}"
    df = pd.read_csv(csv_path)
    os.makedirs(f"inference_results/{output_dir}/plots", exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.2)
    # agg(): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html
    # MSE
    summary = df.groupby("Folder").agg(
        MSE_mean=("MSE", "mean"),
        MSE_min=("MSE", "min"),
        MSE_max=("MSE", "max")
    ).reset_index()

    # For error bars
    summary["MSE_err_low"] = summary["MSE_mean"] - summary["MSE_min"]
    summary["MSE_err_high"] = summary["MSE_max"] - summary["MSE_mean"]
    plt.figure(figsize=(12, 7.5))

    sns.barplot(
        data=summary,
        x="Folder",
        y="MSE_mean",
        yerr=[summary["MSE_err_low"], summary["MSE_err_high"]],
        color="orange",
        capsize=0.1
    )
    plt.xlabel("Folder")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("MSE by Folder (min, mean, max)")
    plt.tight_layout()
    plt.savefig(f"inference_results/{output_dir}/plots/MSE_by_folder.svg")
    plt.savefig(f"inference_results/{output_dir}/plots/MSE_by_folder.pdf")
    plt.savefig(f"inference_results/{output_dir}/plots/MSE_by_folder.png", dpi=300)
    plt.close()

    # PSNR
    summary = df.groupby("Folder").agg(
        PSNR_mean=("PSNR", "mean"),
        PSNR_min=("PSNR", "min"),
        PSNR_max=("PSNR", "max")
    ).reset_index()

    summary["PSNR_err_low"] = summary["PSNR_mean"] - summary["PSNR_min"]
    summary["PSNR_err_high"] = summary["PSNR_max"] - summary["PSNR_mean"]
    plt.figure(figsize=(12, 7.5))

    sns.barplot(
        data=summary,
        x="Folder",
        y="PSNR_mean",
        yerr=[summary["PSNR_err_low"], summary["PSNR_err_high"]],
        color="blue",
        capsize=0.1
    )
    plt.xlabel("Folder")
    plt.ylabel("Peak Signal-to-Noise Ratio (PSNR), dB")
    plt.title("PSNR by Folder (min, mean, max)")
    plt.tight_layout()
    plt.savefig(f"inference_results/{output_dir}/plots/PSNR_by_folder.svg")
    plt.savefig(f"inference_results/{output_dir}/plots/PSNR_by_folder.pdf")
    plt.savefig(f"inference_results/{output_dir}/plots/PSNR_by_folder.png", dpi=300)
    plt.close()

    # SSIM
    summary = df.groupby("Folder").agg(
        SSIM_mean=("SSIM", "mean"),
        SSIM_min=("SSIM", "min"),
        SSIM_max=("SSIM", "max")
    ).reset_index()

    summary["SSIM_err_low"] = summary["SSIM_mean"] - summary["SSIM_min"]
    summary["SSIM_err_high"] = summary["SSIM_max"] - summary["SSIM_mean"]
    plt.figure(figsize=(12, 7.5))

    sns.barplot(
        data=summary,
        x="Folder",
        y="SSIM_mean",
        yerr=[summary["SSIM_err_low"], summary["SSIM_err_high"]],
        color="green",
        capsize=0.1
    )
    plt.xlabel("Folder")
    plt.ylabel("Mean Squared Error (SSIM)")
    plt.title("SSIM by Folder (min, mean, max)")
    plt.tight_layout()
    plt.savefig(f"inference_results/{output_dir}/plots/SSIM_by_folder.svg")
    plt.savefig(f"inference_results/{output_dir}/plots/SSIM_by_folder.pdf")
    plt.savefig(f"inference_results/{output_dir}/plots/SSIM_by_folder.png", dpi=300)
    plt.close()    
    return


# Old implementation
'''
def cal_MSE(composite_image, ground_truth, mask):
    # Consider masked regions only, pick all channels for each pixel location where mask==0
    comp_inpainted = composite_image[:, mask==0]
    gt_inpainted = ground_truth[:, mask==0]
    MSE = np.mean((comp_inpainted - gt_inpainted)**2)
    if MSE < 1e-10:
        return 100.0  # Very identical, see James' dynamicenv.py, line 19
    return MSE

# Calculate peak signal to noise ratio (PSNR) over inpainted region, returns scalar value in decbels, dB
# Formula: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
def cal_PSNR(composite_image, ground_truth, mask):
    data_range = ground_truth.max() - ground_truth.min()
    MSE = cal_MSE(composite_image, ground_truth, mask)
    PSNR = 10 * np.log10((data_range)**2 / MSE)
    return PSNR

# Calculate mean structural similarity index (SSIM) over inpainted region
def cal_SSIM(composite_image, ground_truth, mask, win_size=7):
    # Find indices that satisfy the condition 'mask==0' (unknown regions)
    # https://www.programiz.com/python-programming/numpy/methods/argwhere
    coords = np.argwhere(mask==0)
    if coords.size == 0:  # No hole hence no inpainting
        return 1.0
    
    # Minimal bounding box for SSIM calculation -> Union of all square masks
    # Compare row (axis = 0)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1  # +1 for slicing
    
    # Debug & Verification
    # print("x0: ", x0)
    # print("x1: ", x1)
    # print("y0: ", y0)
    # print("y1: ", y1)

    # Crop -> (3, H, W)
    comp_crop = composite_image[:, y0:y1, x0:x1]
    gt_crop = ground_truth[:, y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]  # Binary/Grayscale

    if comp_crop.size == 0:  # No inpainting needed
        return 1.0
    
    # Don't calculate SSIM if region is too small to judge (<7x7)
    _, h_cropped, w_cropped = comp_crop.shape
    if h_cropped < win_size or w_cropped < win_size:
        return 1.0 

    data_range = gt_crop.max() - gt_crop.min()

    # Zero out known regions for all 3 channels so SSIM ignores them for structural comparison
    comp_crop[:, mask_crop==1] = 0.0
    gt_crop[:, mask_crop==1] = 0.0

    # Syntax: https://scikit-image.org/docs/0.24.x/api/skimage.metrics.html#skimage.metrics.structural_similarity
    # Image shape: (C, H, W) -> Channel axis = 0
    SSIM_value = SSIM(gt_crop, comp_crop, data_range=data_range, channel_axis=0, win_size=win_size)
    return SSIM_value
'''