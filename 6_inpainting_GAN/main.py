import os
import glob
import argparse
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset


# -------------------------------------------------------
# 1) Dataset for 128x128 GRAYSCALE ellipse images
# -------------------------------------------------------
class EllipseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("L")  # Grayscale
        if self.transform:
            img = self.transform(img)
        return img


# -------------------------------------------------------
# 2) Random Square Mask Generation
# -------------------------------------------------------
def create_random_mask(height=128, width=128, coverage=0.3):
    """
    Generates a (1, H, W) mask: 1=valid pixel, 0=missing pixel.
    The missing region is a random square occupying ~'coverage' fraction.
    """
    mask = np.ones((1, height, width), dtype=np.float32)
    mask_area = int(coverage * height * width)
    side_length = max(int(np.sqrt(mask_area)), 1)
    half_side = side_length // 2
    center_x = np.random.randint(half_side, height - half_side)
    center_y = np.random.randint(half_side, width - half_side)
    start_x = center_x - half_side
    end_x = start_x + side_length
    start_y = center_y - half_side
    end_y = start_y + side_length
    mask[0, start_x:end_x, start_y:end_y] = 0.0
    return mask


def apply_random_mask(images, coverage=0.3):
    """
    images: (B,1,128,128) in [-1,1]
    Returns:
        masked_imgs: replaced missing region with -1
        masks: binary 1/0 mask
    """
    B, _, H, W = images.shape
    masked_list = []
    mask_list = []
    device = images.device

    for i in range(B):
        single_mask = create_random_mask(H, W, coverage)
        single_mask_tensor = torch.from_numpy(single_mask).to(device)

        # Where mask=0, set image to -1.0
        masked_img = images[i] * single_mask_tensor + (1 - single_mask_tensor) * (-1.0)
        masked_list.append(masked_img.unsqueeze(0))
        mask_list.append(single_mask_tensor.unsqueeze(0))

    masked_imgs = torch.cat(masked_list, dim=0)
    masks = torch.cat(mask_list, dim=0)
    return masked_imgs, masks


# -------------------------------------------------------
# 3) DCGAN-Style Generator (Grayscale)
# -------------------------------------------------------
class Generator(nn.Module):
    """
    Input: (B,1,128,128) with masked region at -1
    Output: (B,1,128,128) in [-1,1]
    """

    def __init__(self):
        super(Generator, self).__init__()
        # Downsample 4 times: 128->64->32->16->8
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),  # 128->64
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),  # 64->32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),  # 32->16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),  # 16->8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Bottleneck: (B,512,8,8)

        # Upsample 4 times: 8->16->32->64->128
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 8->16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16->32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh()  # 64->128
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        # Decoder
        y1 = self.dec1(x4)
        y2 = self.dec2(y1)
        y3 = self.dec3(y2)
        y4 = self.dec4(y3)
        return y4


# -------------------------------------------------------
# 4) DCGAN-Style Discriminator (Grayscale)
# -------------------------------------------------------
class Discriminator(nn.Module):
    """
    Input: (B,1,128,128), Output: scalar for each sample
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        # 128->64->32->16->8->4-> maybe 1
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),  # 128->64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 64->32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 32->16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),  # 16->8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 2, 1),  # 8->4
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.main(x)  # shape: (B,1,4,4)
        # Flatten to (B,16)
        out = out.view(out.size(0), -1)
        # Option 1: take the mean across the 16 values -> shape (B,)
        validity = out.mean(dim=1)
        return validity


# -------------------------------------------------------
# 5) Weight Initialization
# -------------------------------------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# -------------------------------------------------------
# 6) Training Script
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset/625_images")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.15,
        help="fraction of pixels to mask out in random square",
    )
    parser.add_argument(
        "--model_out", type=str, default="models/inpainting_gan_gray.pth"
    )
    args = parser.parse_args()

    os.makedirs("result/train/real", exist_ok=True)
    os.makedirs("result/train/masked", exist_ok=True)
    os.makedirs("result/train/recon", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Data transform: grayscale in [-1,1]
    transform = T.Compose(
        [T.Resize((128, 128)), T.ToTensor(), T.Normalize((0.5,), (0.5,))]
    )
    dataset = EllipseDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize models
    netG = Generator().to(device)
    netG.apply(weights_init)

    netD = Discriminator().to(device)
    netD.apply(weights_init)

    # Losses
    criterion_bce = nn.BCELoss().to(device)
    criterion_l2 = nn.MSELoss().to(device)

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    real_label = 1.0
    fake_label = 0.0

    for epoch in range(args.epochs):
        for i, real_imgs in enumerate(dataloader):
            real_imgs = real_imgs.to(device)  # shape: (B,1,128,128)
            batch_size = real_imgs.size(0)

            # Step 1: Create masked input and mask
            masked_imgs, masks = apply_random_mask(real_imgs, coverage=args.coverage)
            # masked_imgs: in [-1,1], missing region is set to -1
            # masks: 1=valid, 0=missing

            # Step 2: Train Discriminator
            netD.zero_grad()

            # (a) Real
            label = torch.full(
                (batch_size,), real_label, dtype=torch.float, device=device
            )
            output_real = netD(real_imgs)  # shape (B,)
            errD_real = criterion_bce(output_real, label)
            errD_real.backward()
            D_x = output_real.mean().item()

            # (b) Fake
            fake_full = netG(masked_imgs)
            # composite: real where mask=1, fake where mask=0
            composite = real_imgs * masks + fake_full * (1 - masks)
            label.fill_(fake_label)
            output_fake = netD(composite.detach())
            errD_fake = criterion_bce(output_fake, label)
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            # Step 3: Train Generator
            netG.zero_grad()
            # (a) Adversarial Loss
            label.fill_(real_label)  # generator wants to fool D
            output_fake_forG = netD(composite)
            errG_adv = criterion_bce(output_fake_forG, label)

            # (b) L2 reconstruction loss on masked region only
            recon_loss = criterion_l2(fake_full * (1 - masks), real_imgs * (1 - masks))
            lambda_l2 = 0.3
            errG = (1 - lambda_l2) * errG_adv + lambda_l2 * recon_loss
            errG.backward()
            D_G_z2 = output_fake_forG.mean().item()
            optimizerG.step()

            # Step 4: Print / Save images
            if i % 50 == 0:
                print(
                    f"[{epoch}/{args.epochs}][{i}/{len(dataloader)}] "
                    f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                    f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}"
                )

                # Save sample images
                vutils.save_image(
                    real_imgs.detach(),
                    f"result/train/real/real_epoch_{epoch:03d}.png",
                    normalize=True,
                )
                vutils.save_image(
                    masked_imgs.detach(),
                    f"result/train/masked/masked_epoch_{epoch:03d}.png",
                    normalize=True,
                )
                vutils.save_image(
                    composite.detach(),
                    f"result/train/recon/recon_epoch_{epoch:03d}.png",
                    normalize=True,
                )

        print(f"Epoch [{epoch+1}/{args.epochs}] completed.")

    # Save generator
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    torch.save(netG.state_dict(), args.model_out)
    print(f"Model saved to '{args.model_out}'")

    # Optional: visualize a final example
    sample = next(iter(dataloader))
    sample = sample.to(device)
    masked_ex, mask_ex = apply_random_mask(sample, coverage=args.coverage)
    with torch.no_grad():
        fake_ex = netG(masked_ex)
    composite_ex = sample * mask_ex + fake_ex * (1 - mask_ex)
    recon_np = 0.5 * (composite_ex.cpu().numpy().squeeze() + 1.0)  # [-1,1] -> [0,1]

    plt.imshow(recon_np[0], cmap="gray")
    plt.title("Example reconstructed image")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
