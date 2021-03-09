import sys
import os

def setup_parameters():
    """
    This function sets up the hyperparameters needed to train the model.
    
    Returns:
    hyperparameters : Dictionary containing the hyperparameters for the model.
    options : Dictionary containing the options for the dataset location, augmentation, etc.
    """

    # Dataset location for training.
    dataset_fp = "/file/path/to/dataset"
    
    # Indicate if LiDAR data is included. If it is then no padding or augmenting prior to training
    # will be done. If you still want padding then enable it in the model.
    use_lidar = True

    # File with previously trained weights. Leave as None if you don't want to load any
    pre_trained_weights = None
    # Starting epoch
    start_epoch = 1
    # Max number of epochs
    epochs = 80
    # Starting learning rate for the Adam optimizer
    learn_rate = 0.001
    # Adjusts the learning rate by (learn_rate / 10) after every lr_change epochs
    lr_change = 25
    # Weighted Cross Entropy (put 1.0 for each class if you don't want to add weights)
    class_weights = [0.5, 1.0]
    # Indicate if you want to augment the training images and labels
    augment = False

    # Size of the batch fed into the model. Model can handle a larger batch size when training.
    # Batch size used during training.
    training_batch_size = 10
    # Batch size used during validation.
    valid_batch_size = 5
    
    # Model's learned parameters (i.e. weights and biases) that achieved the lowest loss. Will be
    # saved as "saved_model.pt". A file called "saved_model_last_epoch.pt" will also be saved with
    # the learned parameters in the last completed epoch.
    saved_model = "saved_model"

    ### MODEL PARAMETERS ###

    # Number of input channels
    in_channels = 4
    # Number of output channels 
    n_classes = 2
    # How deep the network will be
    depth = 7
    # Number of filters in the first layer (2**wf)
    wf = 6
    # Indicate if you want the model to pad the images back to their original dimensions
    # Images need to be 256x256 if this is set to False
    pad = True
    # Specify if you want to enable batch normalization
    batch_norm = True
    # Supported modes are 'upconv' and 'upsample'
    up_mode = 'upconv'

    # Store the options in a dictionary
    options = {
            'pre_trained_weights': pre_trained_weights,
            'start_epoch': start_epoch,
            'dataset_fp': dataset_fp,
            'saved_model': saved_model,
            'in_channels': in_channels,
            'n_classes': n_classes,
            'augment': augment,
            'use_lidar': use_lidar
        }
    # Store the hyperparameters in a dictionary
    hyperparameters = {
                    'epochs': epochs,
                    'learn_rate': learn_rate,
                    'lr_change': lr_change,
                    'class_weights': class_weights,
                    'training_batch_size': training_batch_size,
                    'valid_batch_size': valid_batch_size,
                    'in_channels': in_channels,
                    'depth': depth,
                    'wf': wf,
                    'pad': pad,
                    'batch_norm': batch_norm,
                    'up_mode': up_mode
                }

    # Make sure the file paths exist
    if pre_trained_weights is not None and not os.path.isfile(pre_trained_weights):
        sys.exit('Error: Pre-trained weights file does not exist')
    if not os.path.isdir(dataset_fp):
        sys.exit('Error: Main file path to the training and validation images does not exist')

    return hyperparameters, options

