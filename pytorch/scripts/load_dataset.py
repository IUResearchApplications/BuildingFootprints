import torch
import os
import random
from torchvision import transforms
import numpy as np
import glob
from PIL import Image

def test_data(images_fp, lidar_fp, use_lidar):
    """
    Grabs the file names of all the test images.

    Parameters:
    -----------
    data_set: File path given by user leading to the main folder with the data

    Returns:
    --------
    img_tensor: List of file names for the images
    orig_dim: Original dimensions of the images
    total_img: Total number of images in the dataset
    """
    print ('Loading test data...', flush = True)
    # Load in the test dataset
    images = glob.glob(os.path.join(images_fp, '*.tif'))
    if use_lidar:
        lidar_data = glob.glob(os.path.join(lidar_fp, '*.txt'))
    else:
        lidar_data = []

    # Sort the dataset
    images.sort()
    lidar_data.sort()

    # Find the original dimensions of the images
    orig_dim = 0
    img = Image.open(images[0])
    orig_dim = np.asarray(img).shape[1]
    img.close()

    # Find the size of the dataset
    total_img = len(images)
    
    print("Test size: {}".format(total_img))

    return images, lidar_data, orig_dim, total_img

def train_data(data_set, use_lidar, current_set):
    """
    Grabs the file names of all the current set's images and labels.

    Parameters:
    -----------
    data_set: File path given by user leading to the main folder with the data
    current_set: Current dataset to load in

    Returns:
    --------
    img_tensor: List of file names for the images
    lab_tensor: List of file names for the labels
    orig_dim: Original dimensions of the images (or labels)
    num_orig_img: Total number of images (or labels) in the dataset
    """
    # Load in the correct dataset
    if current_set == 'validation':
        print ('Loading {} data...'.format(current_set), flush = True)
        images = glob.glob(os.path.join(data_set, 'validation_images', '*.tif'))
        labels = glob.glob(os.path.join(data_set, 'validation_labels', '*.tif'))
        if use_lidar:
            lidar_data = glob.glob(os.path.join(data_set, 'lidar', 'lidar_validation_data', '*.txt'))
        else:
            lidar_data = []

    else:
        print ('Loading {} data...'.format(current_set), flush = True)
        images = glob.glob(os.path.join(data_set, 'training_images', '*.tif'))
        labels = glob.glob(os.path.join(data_set, 'training_labels', '*.tif'))
        if use_lidar:
            lidar_data = glob.glob(os.path.join(data_set, 'lidar', 'lidar_training_data', '*.txt'))
        else:
            lidar_data= []

    # Sort the datasets
    images.sort()
    labels.sort()
    lidar_data.sort()
    
    img = Image.open(images[0])
    orig_dim = np.asarray(img).shape[1]
    img.close()
    
    # Find the size of the dataset
    num_orig_img = len(images)
    
    print("{} size: {}".format(current_set, num_orig_img))

    return images, labels, lidar_data, orig_dim, num_orig_img
    
