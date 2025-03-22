import pygame
import numpy as np
import pandas as pd
import torchvision.transforms as T
import matplotlib.pyplot as plt

from PIL import Image
from skimage.metrics import structural_similarity as SSIM

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

def cal_PSNR(composite_image, ground_truth, mask):
    comp_inpainted = composite_image[:, mask==0]
    gt_inpainted = ground_truth[:, mask==0]

    data_range = gt_inpainted.max() - gt_inpainted.min()

    MSE = np.mean((comp_inpainted - gt_inpainted)**2)
    if MSE < 1e-10:
        return 100.0
    
    PSNR = 10 * np.log10((data_range)**2 / MSE)
    return PSNR

def cal_SSIM(composite_image, ground_truth, mask):
    coords = np.argwhere(mask==0)

    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    
    comp_crop = composite_image[:, y0:y1, x0:x1]
    gt_crop = ground_truth[:, y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]

    if comp_crop.size == 0:
        return 1.0

    data_range = gt_crop.max() - gt_crop.min()

    comp_crop[:, mask_crop==1] = 0.0
    gt_crop[:, mask_crop==1] = 0.0

    (h_cropped, w_cropped) = comp_crop.shape[-2:]
    min_dim = min(h_cropped, w_cropped)
    win_size = 7
    if min_dim < 7:
        possible_sizes = [x for x in [5,3,1] if x <= min_dim]
        win_size = possible_sizes[0] if possible_sizes else 1
    SSIM_value = SSIM(gt_crop, comp_crop, data_range = data_range, channel_axis = 0, win_size=win_size)
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
    return