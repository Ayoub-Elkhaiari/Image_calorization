import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt


from model import UNet
from my_dataset import ImageColorizationDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

bw_folder = '/content/drive/MyDrive/Data/Black_White'
color_folder = '/content/drive/MyDrive/Data/colored'
dataset = ImageColorizationDataset(bw_folder, color_folder)

# DataLoader for batching
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# Initialize the model and move to device
model = UNet().to(device)

# Loss function and optimizer
criterion = nn.MSELoss()  # Mean squared error for colorization task
optimizer = optim.Adam(model.parameters(), lr=1e-4)



# Training loop
def train(model, dataloader, criterion, optimizer, device, epochs=50):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (bw_images, color_images) in enumerate(dataloader):
            bw_images, color_images = bw_images.to(device), color_images.to(device)

            # Forward pass
            outputs = model(bw_images)
            loss = criterion(outputs, color_images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")

# Start training
train(model, dataloader, criterion, optimizer, device)

# Save the trained model
# torch.save(model.state_dict(), 'unet_colorization.pth')