import glob
import random
import sys
import os
import pygame
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from skimage.metrics import structural_similarity as SSIM

def random_environment(env_img_path, sample):
    imgs_in_folder = sorted(glob.glob(os.path.join(env_img_path, "**", "*.png"), recursive=True))
    print(f"Found {len(imgs_in_folder)} images in '{env_img_path}'")
    return random.sample(imgs_in_folder, sample)

# Scales image according to provided 'img_scaled_dim'
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

# Rotate image -> Resize -> Optionally convert to NumPy array/tensor and normalize to [-1, 1] range
def convert_img_to_(img_path, scaled_dim, output=None, rotation=0):
    img = Image.open(img_path).convert("RGB")
    img = T.functional.rotate(img, rotation)
    transform = T.Compose([ImageResize(scaled_dim=scaled_dim)])
    img = transform(img)

    if output == "nparray":
        return np.array(img)
    
    if output == "tensor":
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        return transform(img)

# Convert NumPy array to Pygame surface
def nparray_to_surface(nparray, scale, grayscale=False):
    if grayscale:
        # Stack thrice, Pygame expects 3 channels
        nparray = np.stack((nparray, nparray, nparray), axis=-1)
    array_scaled = np.kron(nparray, np.ones((scale, scale, 1)))

    # NumPy -> (H, W, C) to  Pygame -> (W, H, 3)
    surface = pygame.surfarray.make_surface(np.transpose(array_scaled, (1, 0, 2)))
    return surface

'''
References:
- https://stackoverflow.com/questions/45393694/size-of-a-dictionary-in-bytes
- https://goshippo.com/blog/measure-real-size-any-python-object
'''
def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

# Tensor shape: (C, H, W), in range [-1, 1]
# Reference: https://pytorch.org/docs/stable/generated/torch.randn_like.html (Gaussian, mean = 0, var = 1)
# 'std' to scale the noise to the desired standard deviation
def add_gaussian_noise_tensor(tensor, std=0.05):
    noise = torch.randn_like(tensor) * std
    noisy = tensor + noise
    return torch.clamp(noisy, -1.0, 1.0)

# Salt-and-pepper noise with 5% coverage
def add_salt_and_pepper_noise_tensor(tensor, amount=0.05):
    noisy = tensor.clone()
    # numel() returns total number of elements in input tensor, e.g., 76800 (320 * 240)
    # Elements per channel
    total_pixels = tensor.numel() // tensor.shape[0]
    # num_salt == num_pepper, num_pepper variable not needed
    num_salt = int(amount * total_pixels / 2)

    # Random locations where noise will be added
    coords = torch.randint(0, total_pixels, (2 * num_salt, ))

    for channel in range(tensor.shape[0]):
        flat = noisy[channel].flatten()
        flat[coords[:num_salt]] = 1.0  # Salt -> White, RGB (255, 255, 255) when normalised later
        flat[coords[num_salt:]] = -1.0  # Pepper -> Black when normalised later

    return torch.clamp(noisy, -1.0, 1.0)

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

def plot_from_csv(output_dir, csv_file="log.csv"):
    csv_path = f"results/{output_dir}/{csv_file}"
    df = pd.read_csv(csv_path)
    os.makedirs(f"results/{output_dir}/plots", exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.2)

    plot_vars = {
        "MSE": "Mean Squared Error (MSE)",
        "PSNR": "Peak Signal-to-Noise Ratio (PSNR), dB",
        "SSIM": "Structural Similarity Index (SSIM)",
        "Percentage Explored": "Percentage Explored (%)"
    }

    for key, label in plot_vars.items():
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df, x="Step", y=key, linewidth=2.0)
        plt.xlabel("Step")
        plt.ylabel(label)
        plt.title(f"{label} vs Step")
        plt.tight_layout()

        for ext in ["png", "svg", "pdf"]:
            plt.savefig(f"results/{output_dir}/plots/{key.replace(' ', '_')}_vs_step.{ext}", dpi=300 if ext == "png" else None)
        plt.close()
    
    return