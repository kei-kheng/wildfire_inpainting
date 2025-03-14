# From 6_inpainting_GAN
import torch.nn as nn


class ContextEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()

        # Encoder
        self.enc = nn.Sequential(
            # Downsample using strided convolutions instead of pooling layers
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.dec = nn.Sequential(
            # Upsample
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Output 3 channels
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Scale images to the range [-1, 1]
        )

    def forward(self, x):
        e = self.enc(x)
        b = self.bottleneck(e)
        out = self.dec(b)
        return out
