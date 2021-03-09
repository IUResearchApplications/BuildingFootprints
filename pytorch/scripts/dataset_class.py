import torch
import random
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image

class CustomDatasetFromTif(Dataset):
    """
    Sets up a custom dataset class for Pytorch's data loader which allows more flexibility
    with loading and setting up the data.
    """
    def __init__ (self, images, labels, lidar_data, total_imgs, orig_dim = None, depth = None,
                  padding = True, augment = False, test_set = False, use_lidar = False):
        """
        images: (# of images, 3, dim, dim) torch tensor containing images from a dataset
        labels: (# of labels, dim, dim) torch tensor containing labels from a datset
        total_imgs: Total images in a dataset
        orig_dim: The images' original dimensions
        depth: How deep the U-Net model is
        padding: Indicates if the model is doing the padding
        augment: Indicates if we should augment the images and labels
        test_set: Indicates if the test set is being loaded in
        use_lidar: Indicates if we are using LiDAR data
        """
        self.total_imgs = total_imgs
        self.images = images
        self.labels = labels
        self.lidar_data = lidar_data
        self.test_set = test_set
        self.augment = augment
        self.use_lidar = use_lidar
        if not padding and not use_lidar:
            self.pad_size = int(((264 - orig_dim + 12 * 2**(depth-1) - 12)) / 2)
            self.pad_label = int((260 - orig_dim) / 2)
        else:
            self.pad_size = 0
            self.pad_label = 0
			
        # Set up the torchvision transforms for non-augmented dataset
        self.img_transform = transforms.Compose([transforms.Pad(self.pad_size),
                                       transforms.ToTensor()])
        self.label_transform = transforms.Compose([transforms.Pad(self.pad_label),
                                         transforms.ToTensor()])

		# Set up the torchvision transforms for augmented dataset
        self.augmentation_img = transforms.Compose([transforms.RandomVerticalFlip(p=0.5),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.Pad(self.pad_size),
                                          transforms.ToTensor()])
        self.augmentation_label = transforms.Compose([transforms.RandomVerticalFlip(p=0.5),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.Pad(self.pad_label),
                                            transforms.ToTensor()])
                                            

    def __getitem__(self, index):
        """
        Loads the next image and label in the batch. Will cycle through the entire dataset.

        Parameters:
        -----------
        index: Tells what image to grab from a torch tensor containing the images and labels

        Returns:
        --------
        self.images[index]: The image 
        self.labels[index]: The label
        """
        
        
        if self.use_lidar:
            # Min-max normalize the lidar data
            lidar_array = np.loadtxt(self.lidar_data[index], delimiter=',').reshape((1, 256, 256))
            lidar_array = (lidar_array - 516.97) / (2510.11 - 516.97)
            lidar_tensor = torch.from_numpy(lidar_array).float()
        
        img = Image.open(self.images[index])

        if self.test_set:
            img_transformed = self.img_transform(img)
            img.close()
            if self.use_lidar:
                img_cat = torch.cat((img_transformed, lidar_tensor), dim=0)
                return img_cat
            
            return img_transformed
        else:
            lab = Image.open(self.labels[index])
            
            if self.augment and not self.use_lidar:
                img_transformed = self.augmentation_img(img)
                lab_transformed = self.augmentation_label(lab)  
            else:
                img_transformed = self.img_transform(img)
                lab_transformed = self.label_transform(lab)
            img.close()
            lab.close()
            if self.use_lidar:
                img_cat = torch.cat((img_transformed, lidar_tensor), dim=0)
                return img_cat, torch.squeeze(lab_transformed)
                
            return img_transformed, torch.squeeze(lab_transformed)

    def __len__(self):
        """
        Returns the total number of images
        """
        return self.total_imgs

