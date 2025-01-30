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
import random


# Data setup
class EllipseDataset(Dataset):
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


def create_random_mask(height=128, width=128, coverage=0.3):
    mask = np.ones((1, height, width), dtype=np.float32)
    num_pixels = height * width
    num_mask = int(num_pixels * coverage)
    coords = np.random.choice(num_pixels, num_mask, replace=False)
    mask_flat = mask.reshape(-1)
    mask_flat[coords] = 0.0
    return mask_flat.reshape(1, height, width)


# inpainting.py in 1_training_with_intact_images had a default batch size of 1, this function deals with batch size >= 1
def apply_random_mask(batch, coverage=0.3):
    # batch.shape returns the batch size, channel (1 for grayscale, 3 for RGB, _ means we do not need to name that variable), height and width
    B, _, H, W = batch.shape
    masked = []
    masks = []
    for i in range(B):
        single_mask = create_random_mask(H, W, coverage)
        single_mask_tensor = torch.from_numpy(single_mask).to(batch.device)
        masked_img = batch[i] * single_mask_tensor
        # Adds batch dimension at index 0 (needed for concatenation later)
        masked.append(masked_img.unsqueeze(0))
        masks.append(single_mask_tensor.unsqueeze(0))
    # Combine the list of tensors into a single tensor along a chosen dimension (specified by dim, the batch dimension in this case)
    # Necessary to tell PyTorch that each item in the list corresponds to a tensor of batch size of 1
    # B will then be the 'new'/'stacked' batch size of the (automatically computed by torch.cat)
    masked = torch.cat(masked, dim=0)  # (B,1,H,W)
    masks = torch.cat(masks, dim=0)  # (B,1,H,W)
    return masked, masks


# Model definition (same as 1_training_with_intact_images) - we are only changing the way we train it - using masked images instead of intact ones
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
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


# Training
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/training",
        help="Path to folder of training images.",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train."
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.3,
        help="Fraction of pixels to set to 0 in mask.",
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default="model/inpainting_autoencoder.pth",
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--num_show", type=int, default=5, help="Number of images to display."
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    print("Testing data directory:", data_dir)

    transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])

    # Uncomment and use this version to avoid overfitting
    # transform = T.Compose([
    #     T.RandomRotation(20),
    #     T.RandomResizedCrop((128,128), scale=(0.8,1.0)),
    #     T.ToTensor()
    # ])

    dataset = EllipseDataset(data_dir, transform=transform)
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

    num_epochs = args.epochs
    coverage = args.coverage

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images in train_loader:
            images = images.to(device)

            # On-the-fly masking
            masked_imgs, _ = apply_random_mask(images, coverage=coverage)

            outputs = model(masked_imgs)  # Forward pass
            loss = criterion(outputs, images)  # Calculate MSE

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
                masked_imgs, _ = apply_random_mask(images, coverage=coverage)
                outputs = model(masked_imgs)
                loss = criterion(outputs, images)
                test_loss += loss.item() * images.size(0)
        test_loss = test_loss / len(test_loader.dataset)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}"
        )

    # Save model
    out_path = os.path.abspath(args.model_out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Model saved to '{out_path}'")

    # Visualisation
    model.eval()
    sample_batch = next(iter(test_loader))
    sample_batch = sample_batch.to(device)

    masked_imgs, _ = apply_random_mask(sample_batch, coverage=coverage)
    with torch.no_grad():
        recon = model(masked_imgs)

    num_show = min(args.num_show, sample_batch.shape[0])
    plt.figure(figsize=(10, 4))
    for i in range(num_show):
        orig = sample_batch[i].cpu().numpy().squeeze()
        mskd = masked_imgs[i].cpu().numpy().squeeze()
        rcn = recon[i].cpu().numpy().squeeze()

        ax1 = plt.subplot(num_show, 3, i * 3 + 1)
        plt.imshow(orig, cmap="gray")
        plt.title("Original")
        plt.axis("off")

        ax2 = plt.subplot(num_show, 3, i * 3 + 2)
        plt.imshow(mskd, cmap="gray")
        plt.title("Masked")
        plt.axis("off")

        ax3 = plt.subplot(num_show, 3, i * 3 + 3)
        plt.imshow(rcn, cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
