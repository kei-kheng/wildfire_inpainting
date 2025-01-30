import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

# Function to generate mask based on specified coverage (fraction)
def create_random_mask(height=128, width=128, coverage=0.3):
    # Create an array of shape (1, height, width) with all values = 1
    mask = np.ones((1, height, width), dtype=np.float32)
    num_pixels = height * width
    num_mask = int(num_pixels * coverage)
    # Randomly pick num_mask unique indices from [0, num_pixels-1], replace = False ensures that the same index is not picked twice
    coords = np.random.choice(num_pixels, num_mask, replace=False)
    # Flatten into a 1D array for easier indexing
    mask_flat = mask.reshape(-1)
    # For those randomly chosen indices, set the values to 0
    # coords is an array of indices
    # NumPy does vectorised operations on arrays without explicit for loops
    mask_flat[coords] = 0.0
    # Reshape and return mask
    return mask_flat.reshape(1, height, width)

def apply_random_mask(img, coverage=0.3):
    # img shape: (1,H,W)
    mask = create_random_mask(img.shape[1], img.shape[2], coverage)
    # Convert the NumPy mask to a PyTorch tensor, then move it to 'device'
    mask_tensor = torch.from_numpy(mask).to(img.device)
    # Wherever mask_tensor is 0, result becomes 0 (black, masked out)
    masked_img = img * mask_tensor
    return masked_img, mask_tensor

# Class that inherits PyTorch's Dataset, same as EllipseDataset from main.py but given a more intuitive name here
class InferenceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img

# Same as main.py
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.enc1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.dec1(x))
        x = torch.sigmoid(self.dec2(x))
        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/testing",
        help="Path to folder of images to do inpainting on.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/autoencoder.pth",
        help="Path to a trained model.",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.3,
        help="Fraction of pixels to set to 0 in mask.",
    )
    parser.add_argument(
        "--num_show", type=int, default=5, help="Number of images to display."
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    print("Testing data directory:", data_dir)
    model_path = os.path.abspath(args.model_path)

    transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
    dataset = InferenceDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Instantiation is still necessary
    model = ConvAutoencoder().to(device)
    '''
    - torch.load() reads a.pth or .pt file that contains the state dict (dictionary of parameter names and weight tensors) of a PyTorch model
    - map_location = device tells PyTorch to map the saved weights to whichever device you specify
    - load_state_dict() takes the state dict and loads them into the model's layers
    '''
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()
    print(f"Loaded model from '{model_path}'")

    results = []
    # loader is an iterator defined above, i indexes current batch, img is the batch (tensor of images)
    # enumerate in Python: https://www.geeksforgeeks.org/enumerate-in-python/
    for i, img in enumerate(loader):
        img = img.to(device)  # shape (1,1,128,128), batch dimension, grayscale channel, height and width

        # Extract first (and only) image from batch, function returns masked image and the mask tensor
        masked_img, mask = apply_random_mask(img[0], coverage=args.coverage)
        # Insert batch dimension at index 0 as model expects something like (1, 1, 128, 128)
        masked_img = masked_img.unsqueeze(0)

        # Forward pass the batch (batch size of 1, see definition of loader above)
        # Don't want gradient tracking because no backpropagation is involved, we want inferencing
        with torch.no_grad():
            output = model(masked_img)

        # Append a tuple of four items to the tuple list
        results.append((img[0].cpu(), masked_img[0].cpu(), output[0].cpu(), mask.cpu()))
        if i >= args.num_show - 1:
            break

    # Set height to 12 inches by 3 *args.num_show inches (width, height)
    plt.figure(figsize=(12, 3 * args.num_show))
    for j, (original, masked, recon, mask) in enumerate(results):
        # Convert tensors to NumPy arrays, remove any singleton dimensions (e.g., (1, 128, 128) to (128, 128))
        # Singleton dimension - dimensions that have length of 1. This give us plain 2D NumPy arrays for plotting
        orig_np = original.numpy().squeeze()
        masked_np = masked.numpy().squeeze()
        recon_np = recon.numpy().squeeze()
        mask_np = mask.numpy().squeeze()

        plt.subplot(args.num_show, 4, j * 4 + 1)
        # Notice differences in this line between main.py and inpainting.py (explained in logbook)
        plt.imshow(orig_np, cmap="gray")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(args.num_show, 4, j * 4 + 2)
        plt.imshow(masked_np, cmap="gray")
        plt.title(f"Masked (cov={args.coverage})")
        plt.axis("off")

        plt.subplot(args.num_show, 4, j * 4 + 3)
        plt.imshow(mask_np, cmap="gray")
        plt.title("Mask")
        plt.axis("off")

        plt.subplot(args.num_show, 4, j * 4 + 4)
        plt.imshow(recon_np, cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")
    
    # Adjust spacing to prevent overlapping
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
