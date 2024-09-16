import os
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from loading import load_images
from model import unet_model



# Paths to the dataset
bw_path = '/content/drive/MyDrive/Data/Black_White'
color_path = '/content/drive/MyDrive/Data/colored'
test_path = '/content/drive/MyDrive/Data/Test'

# Load grayscale and color images and ensure they match in number
bw_images = load_images(bw_path, is_grayscale=True)
color_images = load_images(color_path)

# Ensure both arrays have the same number of images
min_len = min(len(bw_images), len(color_images))
bw_images = bw_images[:min_len]
color_images = color_images[:min_len]

# Normalize images to range [0, 1]
bw_images = bw_images / 255.0
color_images = color_images / 255.0

# print(f'Grayscale images shape: {bw_images.shape}')
# print(f'Color images shape: {color_images.shape}')


unet = unet_model()
unet.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
history = unet.fit(bw_images, color_images, epochs=50, batch_size=8, validation_split=0.1)



# Plot training history
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# # Save the model
# unet.save('unet_colorization.h5')
