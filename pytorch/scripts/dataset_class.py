import torch
import random
from torch.utils.data.dataset import Dataset

class CustomDatasetFromTif(Dataset):
    """
    Sets up a custom dataset class for Pytorch's data loader which allows more flexibility
    with loading and setting up the data.
    """
    def __init__ (self, images, labels, total_imgs, test_set = False, shuffle_data = True):
        """
        images: (# of images, 3, dim, dim) torch tensor containing images from a dataset
        labels: (# of labels, dim, dim) torch tensor containing labels from a datset
        total_imgs: Total images in a dataset
        test_set: Indicates if the test set is being loaded in
        shuffle_data: Indicates if the data should be shuffled
        """
        self.total_imgs = total_imgs
        self.images = images
        self.labels = labels
        self.test_set = test_set

        if shuffle_data:
            # Shuffle the dataset
            shuff = list(zip(images, labels))
            random.shuffle(shuff)
            self.images, self.labels = zip(*shuff)

    def __getitem__(self, index):
        """
        Grabs the next image and label in the batch. Will cycle through the entire dataset.

        Parameters:
        -----------
        index: Tells what image to grab from a torch tensor containing the images and labels

        Returns:
        --------
        self.images[index]: The image 
        self.labels[index]: The label
        """

        if self.test_set:
            return self.images[index]

        else:
            return self.images[index], self.labels[index]

    def __len__(self):
        """
        Returns the number of training images
        """
        return self.total_imgs

