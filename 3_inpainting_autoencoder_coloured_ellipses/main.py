import os
import glob
import argparse
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


# Data setup
class ColorEllipseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        # We want to inpaint RGB images now
        # Reference: https://www.geeksforgeeks.org/python-pil-image-convert-method/
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def create_random_mask(height, width, coverage=0.3):
    # Create a single-channel mask (planar, 2D) and replicate it for two other layers
    # If the 3 channels (RGB) are masked independently, might end up with partial colour information at each pixel (does not make physical sense)
    mask_1ch = np.ones((height, width), dtype=np.float32)
    num_pixels = height * width
    num_mask = int(num_pixels * coverage)
    coords = np.random.choice(num_pixels, num_mask, replace=False)
    mask_1ch_flat = mask_1ch.reshape(-1)
    mask_1ch_flat[coords] = 0.0
    # Add a new dimension at index 0 of mask_1ch => (1, H, W)
    # Repeats the (1, H, W) array 3 times along the dimension at index 0 (channel)
    # Reference: https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
    mask_3ch = np.repeat(mask_1ch[np.newaxis, :, :], 3, axis=0)
    # Shape: (3, H, W)
    return mask_3ch


def apply_random_mask(batch, coverage=0.3):
    B, _, H, W = batch.shape
    masked_list = []
    mask_list = []
    for i in range(B):
        mask_3ch = create_random_mask(H, W, coverage)
        mask_tensor = torch.from_numpy(mask_3ch).to(batch.device)
        masked_img = batch[i] * mask_tensor
        masked_list.append(masked_img.unsqueeze(0))
        mask_list.append(mask_tensor.unsqueeze(0))

    masked = torch.cat(masked_list, dim=0)  # (B,3,H,W)
    masks = torch.cat(mask_list, dim=0)  # (B,3,H,W)
    return masked, masks


# Try with same CAE architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2)

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
    parser.add_argument("--data_dir", type=str, default="dataset/training")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--coverage", type=float, default=0.3)
    parser.add_argument(
        "--model_out", type=str, default="model/inpainting_color_autoencoder.pth"
    )
    parser.add_argument("--num_show", type=int, default=5)
    args = parser.parse_args()

    transform = T.Compose([T.Resize((720, 720)), T.ToTensor()])
    dataset = ColorEllipseDataset(args.data_dir, transform=transform)
    print("Dataset length:", len(dataset))

    # Split data into training set (90%) and testing set (10%, never used for training)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images in train_loader:
            images = images.to(device)

            masked_imgs, _ = apply_random_mask(images, coverage=args.coverage)

            outputs = model(masked_imgs)
            # Unlike previous case (grayscale images), this calculates the MSE for 3 channels
            # Reference: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Evaluate
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)
                masked_imgs, _ = apply_random_mask(images, coverage=args.coverage)
                outputs = model(masked_imgs)
                loss = criterion(outputs, images)
                test_loss += loss.item() * images.size(0)
        test_loss /= len(test_loader.dataset)

        print(
            f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}"
        )

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    torch.save(model.state_dict(), args.model_out)
    print(f"Model saved to '{args.model_out}'")

    # Visualisation
    model.eval()
    sample_batch = next(iter(test_loader))
    sample_batch = sample_batch.to(device)

    masked_imgs, _ = apply_random_mask(sample_batch, coverage=args.coverage)
    with torch.no_grad():
        recon = model(masked_imgs)

    num_show = min(args.num_show, sample_batch.size(0))
    plt.figure(figsize=(12, 4 * num_show))
    for i in range(num_show):
        # Tensor is (3, H, W), but imshow() expects (H, W, 3)
        orig = sample_batch[i].cpu().permute(1, 2, 0).numpy()
        mskd = masked_imgs[i].cpu().permute(1, 2, 0).numpy()
        rcn = recon[i].cpu().permute(1, 2, 0).numpy()

        ax1 = plt.subplot(num_show, 3, i * 3 + 1)
        # No cmap to show true RGB
        plt.imshow(orig)
        plt.axis("off")
        plt.title("Original")

        ax2 = plt.subplot(num_show, 3, i * 3 + 2)
        plt.imshow(mskd)
        plt.axis("off")
        plt.title("Masked")

        ax3 = plt.subplot(num_show, 3, i * 3 + 3)
        plt.imshow(rcn)
        plt.axis("off")
        plt.title("Reconstructed")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
