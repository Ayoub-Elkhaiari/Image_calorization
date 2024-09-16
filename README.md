# Image Colorization Using U-Net

## Overview
This project implements an image colorization model using the U-Net architecture. The model takes grayscale images as input and generates corresponding colorized images. The project includes implementations in both PyTorch and TensorFlow, allowing users to choose their preferred framework.

## Project Structure
- `model.py`: Contains the U-Net model architecture (PyTorch version).
- `my_dataset.py`: Defines the custom dataset class for loading and preprocessing images.
- `main.py`: The main script for training the model (PyTorch version).
- `tensorflow_version/`: Directory containing the TensorFlow implementation (not detailed in the provided code).
- `Data/`: Directory containing the dataset
  - `Black_White/`: Folder with grayscale images
  - `colored/`: Folder with corresponding color images
  - `test/`: Folder with test images

## Prerequisites
- Python 3.x
- PyTorch
- TensorFlow (for TensorFlow version)
- OpenCV (cv2)
- NumPy
- Matplotlib

To install the necessary packages, run:
```bash
pip install torch torchvision tensorflow opencv-python numpy matplotlib
```

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Ayoub-Elkhaiari/Image_calorization.git
   cd Image_calorization
   ```

2. Prepare your dataset:
   - Place grayscale images in the `Data/Black_White/` folder.
   - Place corresponding color images in the `Data/colored/` folder.
   - (Optional) Add test images to the `Data/test/` folder.

## Usage

### Training the Model (PyTorch version)
1. Adjust the data paths in `main.py` if necessary:
   ```python
   bw_folder = 'Data/Black_White'
   color_folder = 'Data/colored'
   ```

2. Run the training script:
   ```bash
   python main.py
   ```

3. The script will train the model and display the loss for each epoch.

4. After training, the model will be saved as `unet_colorization.pth`.

### Using the TensorFlow Version
The project also includes a TensorFlow implementation, which is similar in structure to the PyTorch version. To use it:

1. Navigate to the TensorFlow directory:
   ```bash
   cd tensorflow_version
   ```

2. Follow similar steps as the PyTorch version, adjusting for TensorFlow-specific syntax and practices.

## Key Components

### U-Net Model (`model.py`)
The U-Net architecture is implemented in PyTorch, consisting of:
- Encoder: Four convolutional blocks with max pooling.
- Bottleneck: A convolutional block at the bottom of the 'U'.
- Decoder: Four up-convolutional blocks with skip connections.
- Final layer: A convolutional layer to produce the colorized output.

### Dataset Class (`my_dataset.py`)
`ImageColorizationDataset` is a custom PyTorch Dataset class that:
- Loads grayscale and color image pairs.
- Resizes images to a specified size (default 256x256).
- Normalizes pixel values to the range [0, 1].

### Training Script (`main.py`)
The main script handles:
- Data loading and preprocessing.
- Model initialization.
- Training loop with loss calculation and optimization.
- Saving the trained model.

## Customization
- Adjust the `img_size` parameter in `ImageColorizationDataset` to change input image dimensions.
- Modify the learning rate or optimizer in `main.py` to experiment with different training configurations.
- Add data augmentation techniques in `my_dataset.py` to improve model generalization.

## Notes
- The model uses Mean Squared Error (MSE) as the loss function for colorization.
- The PyTorch implementation uses the Adam optimizer with a learning rate of 1e-4.
- Ensure your GPU drivers and CUDA are properly set up if you intend to use GPU acceleration.

## Future Improvements
- Implement a validation loop to monitor overfitting.
- Add a script for colorizing new grayscale images using the trained model.
- Experiment with different loss functions or additional perceptual losses.
- Implement data augmentation to improve model generalization.

