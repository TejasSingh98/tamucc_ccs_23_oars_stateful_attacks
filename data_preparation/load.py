import pickle
import os
import numpy as np
from PIL import Image

def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Example file path to one of the CIFAR-10 batch files
batch_file = './cifar-10-batches-py/data_batch_1'
batch_data = load_cifar10_batch(batch_file)

def save_images_from_batch(batch_data, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(len(batch_data[b'data'])):
        # Extract image data
        image_array = batch_data[b'data'][i]
        image_array = image_array.reshape((3, 32, 32)).transpose(1, 2, 0)
        image = Image.fromarray(image_array)

        # Define the file path
        label = batch_data[b'labels'][i]
        filename = f'image_{i}_label_{label}.png'
        image_path = os.path.join(save_dir, filename)

        # Save the image
        image.save(image_path)

# Save images from the loaded batch
save_images_from_batch(batch_data, 'cifar10_images')

