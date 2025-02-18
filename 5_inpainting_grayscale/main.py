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


def apply_random_mask(batch, coverage=0.3):
    B, _, H, W = batch.shape
    masked = []
    masks = []
    for i in range(B):
        single_mask = create_random_mask(H, W, coverage)
        single_mask_tensor = torch.from_numpy(single_mask).to(batch.device)
        masked_img = batch[i] * single_mask_tensor
        masked.append(masked_img.unsqueeze(0))
        masks.append(single_mask_tensor.unsqueeze(0))
    masked = torch.cat(masked, dim=0)  # (B,1,H,W)
    masks = torch.cat(masks, dim=0)  # (B,1,H,W)
    return masked, masks


# An alternative (more intuitive) way to define the CAE, nn.Sequential() is just a sequential container
# Reference: https://www.geeksforgeeks.org/implement-convolutional-autoencoder-in-pytorch-with-cuda/
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Training
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

    transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
    dataset = EllipseDataset(args.data_dir, transform=transform)
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
                masked_imgs, _ = apply_random_mask(images, coverage=args.coverage)
                outputs = model(masked_imgs)
                loss = criterion(outputs, images)
                test_loss += loss.item() * images.size(0)
        test_loss /= len(test_loader.dataset)

        print(
            f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}"
        )

    # Save model
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

    num_show = min(args.num_show, sample_batch.shape[0])
    plt.figure(figsize=(10, 4))
    for i in range(num_show):
        orig = sample_batch[i].cpu().numpy().squeeze()
        mskd = masked_imgs[i].cpu().numpy().squeeze()
        rcn = recon[i].cpu().numpy().squeeze()

        ax1 = plt.subplot(num_show, 3, i * 3 + 1)
        plt.imshow(orig, cmap="gray")
        plt.axis("off")

        ax2 = plt.subplot(num_show, 3, i * 3 + 2)
        plt.imshow(mskd, cmap="gray")
        plt.axis("off")

        ax3 = plt.subplot(num_show, 3, i * 3 + 3)
        plt.imshow(rcn, cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
