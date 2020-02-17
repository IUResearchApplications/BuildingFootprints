import torch
import os
import random
from torchvision import transforms
import numpy as np
import glob
from PIL import Image

def test_data(data_set, depth, padding):
    """
    Loads in the images, preprocesses them, then returns torch tensors containing them

    Parameters:
    -----------
    data_set: File path given by user leading to the main folder with the data
    depth: How deep the network is (used to figure out how much padding to do)
    padding: Boolean to indicate if the program needs to pad beforehand

    Returns:
    --------
    img_tensor: (# of images, 3, dim, dim) Torch tensor of the preprocessed images
    orig_dim: Original dimensions of the images
    total_img: Total number of images in the dataset
    pad_size: Amount of paddng done in each dimension
    """
    print ('Loading test data...')
    # Load in the test dataset
    images = glob.glob(os.path.join(data_set, 'test_images', '*.tif'))

    # Sort the datasets
    images.sort()

    # Find the original dimensions of the images
    orig_dim = 0
    img = Image.open(images[0])
    orig_dim = np.asarray(img).shape[1]
    img.close()

    # If the model is not padding the image we need to do the padding ourselves beforehand
    # this is for recieving an output of dimension 260 for any depth given for a (256, 256) image
    if not padding:
        pad_size = int(((264 - orig_dim + 12 * 2**(depth-1) - 12)) / 2)
    else:
        pad_size = 0

    # Set up the torchvision transforms for non-augmented dataset
    img_transform = transforms.Compose([transforms.Pad(pad_size),
                                       transforms.ToTensor()])

    # Find the size of the dataset
    total_img = len(images)

    # Set up the torch tensors that the images and labels will be stored in
    img_tensor = torch.zeros(total_img, 3, orig_dim + 2 * pad_size, orig_dim + 2 * pad_size)

    # Load in all of the data and transform it however necessary
    for i in range(total_img):
        img = Image.open(images[i])
        img_transformed = img_transform(img)

        img_tensor[i] = img_transformed

        img.close()

    return img_tensor, orig_dim, total_img, pad_size

def train_data(data_set, depth, padding, augment, current_set):
    """
    Loads in the images and labels, preprocesses them, then returns torch tensors containing them

    Parameters:
    -----------
    data_set: File path given by user leading to the main folder with the data
    depth: How deep the network is (used to figure out how much padding to do)
    padding: Boolean to indicate if the program needs to pad beforehand
    augment: Boolean to indicate if the images and labels should be augmented
    current_set: Current dataset to load in

    Returns:
    --------
    img_tensor: (# of images, 3, dim, dim) Torch tensor of the preprocessed images
    lab_tensor: (# of labels, dim, dim) Torch tensor of the preprocessed labels
    orig_dim: Original dimensions of the images (or labels)
    total_img: Total number of images (or labels) in the dataset
    pad_size: Amount of paddng done in each dimension
    """
    # Load in the correct dataset
    if current_set == 'validation':
        print ('Loading {} data...'.format(current_set), flush = True)
        augment = False
        images = glob.glob(os.path.join(data_set, 'validation_images', '*.tif'))
        labels = glob.glob(os.path.join(data_set, 'validation_labels', '*.tif'))

    else:
        print ('Loading {} data...'.format(current_set), flush = True)
        images = glob.glob(os.path.join(data_set, 'training_images', '*.tif'))
        labels = glob.glob(os.path.join(data_set, 'training_labels', '*.tif'))

    # Sort the datasets
    images.sort()
    labels.sort()

    # Find the original dimensions of the images
    orig_dim = 0
    img = Image.open(images[0])
    orig_dim = np.asarray(img).shape[1]
    img.close()

    # If the model is not padding the image we need to do the padding ourselves beforehand
    # this is for recieving an output of dimension 260 for any depth given for a (256, 256) image
    if not padding:
        pad_size = int(((264 - orig_dim + 12 * 2**(depth-1) - 12)) / 2)
        pad_label = int((260 - orig_dim) / 2)
    else:
        pad_size = 0
        pad_label = 0

    # Set up the torchvision transforms for non-augmented dataset
    img_transform = transforms.Compose([transforms.Pad(pad_size),
                                       transforms.ToTensor()])
    label_transform = transforms.Compose([transforms.Pad(pad_label),
                                         transforms.ToTensor()])

    # Set up the torchvision transforms for augmented dataset
    augmentation_img = transforms.Compose([transforms.RandomVerticalFlip(p=0.5),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.Pad(pad_size),
                                          transforms.ToTensor()])
    augmentation_label = transforms.Compose([transforms.RandomVerticalFlip(p=0.5),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.Pad(pad_label),
                                            transforms.ToTensor()])


    # Find the size of the dataset
    num_orig_img = len(images)

    # Check to see if augmentation is on
    num_augmented_img = 1
    if augment:
        # Add augmented images (and labels) for each original image (and label) to the dataset
        num_augmented_img = 2
        print ("Dataset is now {}x bigger".format(num_augmented_img))

    # Increase the size of the dataset accordingly
    total_img = num_orig_img * num_augmented_img

    # Set up the torch tensors that the images and labels will be stored in
    img_tensor = torch.zeros(total_img, 3, orig_dim + 2 * pad_size, orig_dim + 2 * pad_size)
    lab_tensor = torch.zeros(total_img, orig_dim + 2 * pad_label, orig_dim + 2 * pad_label)

    # Keep track of where we are at in the original dataset (relevant for augmentation)
    count = 0

    # Load in all of the data and transform it however necessary
    for i in range(num_orig_img):
        img = Image.open(images[i])
        img_transformed = img_transform(img)
        lab = Image.open(labels[i])
        lab_transformed = label_transform(lab)

        img_tensor[count] = img_transformed
        lab_tensor[count] = lab_transformed

        img.close()
        lab.close()

        count += 1

        # If there is augmentation then add those in too
        for j in range (num_augmented_img - 1):
            # Make a seed with numpy generator
            seed = np.random.randint(12345678)

            # Apply the seed
            random.seed(seed)

            img = Image.open(images[i])
            img_transformed = augmentation_img(img)

            # Apply the seed again so the label is augmented the same way
            random.seed(seed)

            lab = Image.open(labels[i])
            lab_transformed = augmentation_label(lab)

            img_tensor[count] = img_transformed
            lab_tensor[count] = lab_transformed

            img.close()
            lab.close()

            count += 1

    return img_tensor, lab_tensor, orig_dim, total_img, pad_size
    
