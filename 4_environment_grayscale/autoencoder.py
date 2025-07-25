import torch
import torch.nn as nn
import torch.nn.functional as F

from partialconv2d import PartialConv2d

class PConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # return_mask=True to return updated mask
        # Downsample 128->64->32->16
        self.enc1 = PartialConv2d(in_channels=1, out_channels=32, 
                                  kernel_size=4, stride=2, padding=1,
                                  multi_channel=False, return_mask=True)
        self.enc2 = PartialConv2d(in_channels=32, out_channels=64, 
                                  kernel_size=4, stride=2, padding=1,
                                  multi_channel=False, return_mask=True)
        self.enc3 = PartialConv2d(in_channels=64, out_channels=128, 
                                  kernel_size=4, stride=2, padding=1,
                                  multi_channel=False, return_mask=True)
    
    def forward(self, x, m):
        x1, m1 = self.enc1(x, m)  
        x1 = F.relu(x1)

        x2, m2 = self.enc2(x1, m1)
        x2 = F.relu(x2)

        x3, m3 = self.enc3(x2, m2)
        x3 = F.relu(x3)

        return x3, m3

# Decoder
class PConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # https://stackoverflow.com/questions/53654310/what-is-the-difference-between-upsampling2d-and-conv2dtranspose-functions-in-ker
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.dec1 = PartialConv2d(in_channels=128, out_channels=64, 
                                  kernel_size=3, stride=1, padding=1,
                                  multi_channel=False, return_mask=True)
        self.dec2 = PartialConv2d(in_channels=64, out_channels=32, 
                                  kernel_size=3, stride=1, padding=1,
                                  multi_channel=False, return_mask=True)
        self.dec3 = PartialConv2d(in_channels=32, out_channels=1, 
                                  kernel_size=3, stride=1, padding=1,
                                  multi_channel=False, return_mask=True)

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