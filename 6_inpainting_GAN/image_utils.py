import os
import glob
import numpy as np

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

# Generate a square mask that is placed randomly on image
def create_random_square_mask(channels, height, width, coverage):
    mask_1ch = np.ones((height, width), dtype=np.float32)  # Single-channel mask

    # Calculate side of square
    total_pixels = height * width
    mask_area = int(coverage * total_pixels)
    side_length = max(int(np.sqrt(mask_area)), 1)
    half_side = side_length // 2

    # Randomly pick a square location
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
        mask = create_random_square_mask(3, H, W, coverage)
        mask_tensor = torch.from_numpy(mask).to(batch.device)
        masked_img = batch[i] * mask_tensor
        masked_list.append(masked_img.unsqueeze(0))
        mask_list.append(mask_tensor.unsqueeze(0))

    masked = torch.cat(masked_list, dim=0)  # (B,3,H,W)
    masks = torch.cat(mask_list, dim=0)  # (B,3,H,W)
    return masked, masks