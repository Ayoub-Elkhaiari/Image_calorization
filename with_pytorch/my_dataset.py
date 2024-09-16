import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt



# Dataset class for loading grayscale and color images
class ImageColorizationDataset(Dataset):
    def __init__(self, bw_folder, color_folder, img_size=(256, 256)):
        self.bw_folder = bw_folder
        self.color_folder = color_folder
        self.img_size = img_size
        self.bw_images = sorted(os.listdir(bw_folder))
        self.color_images = sorted(os.listdir(color_folder))
        self.min_len = min(len(self.bw_images), len(self.color_images))  # Ensure both have same number of images

    def __len__(self):
        return self.min_len

    def __getitem__(self, idx):
        bw_img_path = os.path.join(self.bw_folder, self.bw_images[idx])
        color_img_path = os.path.join(self.color_folder, self.color_images[idx])
        
        # Load grayscale image (1 channel)
        bw_img = cv2.imread(bw_img_path, cv2.IMREAD_GRAYSCALE)
        bw_img = cv2.resize(bw_img, self.img_size)
        bw_img = np.expand_dims(bw_img, axis=0)  # Add channel dimension

        # Load color image (3 channels)
        color_img = cv2.imread(color_img_path, cv2.IMREAD_COLOR)
        color_img = cv2.resize(color_img, self.img_size)
        color_img = np.transpose(color_img, (2, 0, 1))  # Convert to C x H x W

        # Normalize images to range [0, 1]
        bw_img = bw_img / 255.0
        color_img = color_img / 255.0

        return torch.FloatTensor(bw_img), torch.FloatTensor(color_img)