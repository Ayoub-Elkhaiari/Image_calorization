import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt


# Define U-Net Model in PyTorch
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU()
            )

        def up_block(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Encoder
        self.enc1 = conv_block(1, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        # Decoder
        self.up1 = up_block(1024, 512)
        self.dec1 = conv_block(1024, 512)
        self.up2 = up_block(512, 256)
        self.dec2 = conv_block(512, 256)
        self.up3 = up_block(256, 128)
        self.dec3 = conv_block(256, 128)
        self.up4 = up_block(128, 64)
        self.dec4 = conv_block(128, 64)

        # Final output layer (3 channels for RGB)
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder
        dec1 = self.dec1(torch.cat((self.up1(bottleneck), enc4), dim=1))
        dec2 = self.dec2(torch.cat((self.up2(dec1), enc3), dim=1))
        dec3 = self.dec3(torch.cat((self.up3(dec2), enc2), dim=1))
        dec4 = self.dec4(torch.cat((self.up4(dec3), enc1), dim=1))

        return torch.sigmoid(self.final(dec4))  # Output with sigmoid activation