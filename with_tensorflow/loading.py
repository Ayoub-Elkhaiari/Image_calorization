import os
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt





def load_images(folder, img_size=(256, 256), is_grayscale=False):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if is_grayscale:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, axis=-1)  # Add channel dimension (1)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, img_size)  # Resize to fixed size
        images.append(img)
    return np.array(images)