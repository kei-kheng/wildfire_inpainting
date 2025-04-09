import torch
import torch.nn as nn
import torch.nn.functional as F
from partialconv2d import PartialConv2d


# Convolutional Autoencoder (CAE)
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.dec1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        return x


# Partial Convolutional Autoencoder (PCAE)
# PCAE - Encoder
class PConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # return_mask=True to return updated mask
        # Downsample 128->64->32->16
        self.enc1 = PartialConv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
            multi_channel=True,
            return_mask=True,
        )
        self.enc2 = PartialConv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            multi_channel=True,
            return_mask=True,
        )
        self.enc3 = PartialConv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            multi_channel=True,
            return_mask=True,
        )

    def forward(self, x, m):
        x1, m1 = self.enc1(x, m)
        x1 = F.relu(x1)

        x2, m2 = self.enc2(x1, m1)
        x2 = F.relu(x2)

        x3, m3 = self.enc3(x2, m2)
        x3 = F.relu(x3)

        return x3, m3


# PCAE - Decoder
class PConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # https://stackoverflow.com/questions/53654310/what-is-the-difference-between-upsampling2d-and-conv2dtranspose-functions-in-ker
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.dec1 = PartialConv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            multi_channel=True,
            return_mask=True,
        )
        self.dec2 = PartialConv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            multi_channel=True,
            return_mask=True,
        )
        self.dec3 = PartialConv2d(
            in_channels=32,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            multi_channel=True,
            return_mask=True,
        )

    def forward(self, x, m):
        x = self.upsample(x)
        m = self.upsample(m)
        x1, m1 = self.dec1(x, m)
        x1 = F.relu(x1)

        x2 = self.upsample(x1)
        m2 = self.upsample(m1)
        x2, m2 = self.dec2(x2, m2)
        x2 = F.relu(x2)

        x3 = self.upsample(x2)
        m3 = self.upsample(m2)
        x3, m3 = self.dec3(x3, m3)
        x3 = torch.sigmoid(x3)

        return x3


class PartialConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PConvEncoder()
        self.decoder = PConvDecoder()

    def forward(self, x, m):
        encoded, updated_mask = self.encoder(x, m)
        out = self.decoder(encoded, updated_mask)
        return out


# Generative Adversarial Network (GAN)
# GAN - Context Encoder
class ContextEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()

        self.enc = nn.Sequential(
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

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        e = self.enc(x)
        b = self.bottleneck(e)
        out = self.dec(b)
        return out


# GAN - Patch Discriminator
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Output shape: (B,1,H/8,W/8)
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.main(x)


# For GAN
def weights_init(layer):
    classname = layer.__class__.__name__
    if "Conv" in classname or "ConvTranspose" in classname:  # Mean = 0, Std = 0.02
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)
