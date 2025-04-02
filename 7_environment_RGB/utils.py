import glob
import random
import os
import pygame
import numpy as np
import pandas as pd
import torchvision.transforms as T
import matplotlib.pyplot as plt

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

def cal_PSNR(composite_image, ground_truth):
    data_range = ground_truth.max() - ground_truth.min()
    MSE = np.mean((composite_image - ground_truth)**2)
    if MSE < 1e-10:
        return 100.0
    PSNR = 10 * np.log10((data_range)**2 / MSE)
    return PSNR

def cal_SSIM(composite_image, ground_truth, win_size=7):
    data_range = ground_truth.max() - ground_truth.min()
    SSIM_value = SSIM(ground_truth, composite_image, data_range=data_range, channel_axis=0, win_size=win_size)
    return SSIM_value

def plot_from_csv(output_dir, csv_file="log.csv"):
    csv_path = f"results/{output_dir}/{csv_file}"
    df = pd.read_csv(csv_path)

    # PSNR
    plt.figure(figsize=(8,5))
    plt.plot(df["Step"], df["PSNR"], color='blue', label="PSNR")
    plt.xlabel("Step")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR over steps")
    plt.grid(True)
    plt.savefig(f"results/{output_dir}/PSNR_vs_step.png", dpi=300)
    plt.show()

    # SSIM
    plt.figure(figsize=(8,5))
    plt.plot(df["Step"], df["SSIM"], label="SSIM", color="green")
    plt.xlabel("Step")
    plt.ylabel("SSIM")
    plt.title("SSIM over steps")
    plt.grid(True)
    plt.savefig(f"results/{output_dir}/SSIM_vs_step.png", dpi=300)
    plt.show()

    # Percentage Explored
    plt.figure(figsize=(8,5))
    plt.plot(df["Step"], df["Percentage Explored"], label="Percentage Explored", color="purple")
    plt.xlabel("Step")
    plt.ylabel("Percentage Explored (%)")
    plt.title("Percentage explored over steps")
    plt.grid(True)
    plt.savefig(f"results/{output_dir}/percentage_explored_vs_step.png", dpi=300)
    plt.show()
    return