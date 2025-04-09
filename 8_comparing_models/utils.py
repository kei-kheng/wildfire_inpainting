import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset
from skimage.metrics import structural_similarity as SSIM


# Dataset
class EllipseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(img_path)


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

        return T.Resize((snapped_h, snapped_w))(img)


def create_random_mask(channels, height, width, coverage):
    mask_1ch = np.ones((height, width), dtype=np.float32)
    mask_area = int(coverage * height * width)
    side_length = max(int(np.sqrt(mask_area)), 1)

    half_side = side_length // 2
    center_x = np.random.randint(half_side, height - half_side)
    center_y = np.random.randint(half_side, width - half_side)

    start_x = center_x - half_side
    end_x = start_x + side_length
    start_y = center_y - half_side
    end_y = start_y + side_length

    mask_1ch[start_x:end_x, start_y:end_y] = 0.0
    mask_3ch = np.repeat(mask_1ch[np.newaxis, :, :], channels, axis=0)
    return mask_3ch


def apply_mask(batch, coverage):
    B, _, H, W = batch.shape
    masked_list = []
    mask_list = []

    for i in range(B):
        mask = create_random_mask(3, H, W, coverage)
        mask_tensor = torch.from_numpy(mask).to(batch.device)
        masked_img = batch[i] * mask_tensor
        masked_list.append(masked_img.unsqueeze(0))
        mask_list.append(mask_tensor.unsqueeze(0))

    masked = torch.cat(masked_list, dim=0)
    masks = torch.cat(mask_list, dim=0)
    return masked, masks


def cal_MSE(composite_image, ground_truth, mask):
    comp_inpainted = composite_image[:, mask == 0]
    gt_inpainted = ground_truth[:, mask == 0]
    MSE = np.mean((comp_inpainted - gt_inpainted) ** 2)
    if MSE < 1e-10:
        return 100.0
    return MSE


def cal_PSNR(composite_image, ground_truth, mask):
    data_range = ground_truth.max() - ground_truth.min()
    MSE = cal_MSE(composite_image, ground_truth, mask)
    PSNR = 10 * np.log10((data_range) ** 2 / MSE)
    return PSNR


def cal_SSIM(composite_image, ground_truth, mask):
    coords = np.argwhere(mask == 0)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1

    comp_crop = composite_image[:, y0:y1, x0:x1]
    gt_crop = ground_truth[:, y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]

    data_range = gt_crop.max() - gt_crop.min()

    comp_crop[:, mask_crop == 1] = 0.0
    gt_crop[:, mask_crop == 1] = 0.0

    SSIM_value = SSIM(
        gt_crop, comp_crop, data_range=data_range, channel_axis=0, win_size=7
    )
    return SSIM_value


def plot_from_csv_training(model_type, output_dir, csv_file="training_log.csv"):
    csv_path = f"results/{model_type}/{output_dir}/{csv_file}"
    df = pd.read_csv(csv_path)
    df_avg = df.groupby("Epoch")[["MSE", "PSNR", "SSIM"]].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df_avg.index, df_avg["MSE"], label="Mean Squared Error (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("MSE per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{model_type}/{output_dir}/MSE_vs_epoch.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df_avg.index, df_avg["PSNR"], label="Peak Signal to Noise Ratio (PSNR)")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{model_type}/{output_dir}/PSNR_vs_epoch.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(
        df_avg.index, df_avg["SSIM"], label="Structural Similarity Index Measure (SSIM)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("SSIM per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{model_type}/{output_dir}/SSIM_vs_epoch.png", dpi=300)
    plt.show()
