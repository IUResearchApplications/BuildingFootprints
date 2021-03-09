import sys
import os

def setup_parameters():
    """
    This function sets up the hyperparameters needed to run test.py. The Model's parameters
    given must match the parameters used when training the model.
    
    Returns:
    hyperparameters : Dictionary containing the hyperparameters for the model
    options : Dictionary containing the options for the dataset location, augmentation, etc.
    """
    # Test images location. The files need to be .tif files.
    images_fp = "/file/path/to/test/images"
    
    # Indicate if LiDAR data is included.
    use_lidar = False
    # LiDAR data location. The files need to be .txt files.
    lidar_fp = "/file/path/to/lidar/data"

    # Batch size used during testing.
    testing_batch_size = 5
    
    # Model's learned parameters (i.e. weights and biases). Should be in saved_models directory. The
    # .pt extension is added in the program.
    saved_model = "saved_model"
    # Main file path to where the predictions are saved (folder 'predictions' will be created).
    predictions = "/file/path/to/save/predictions/at"

    ### MODEL PARAMETERS ###

    # Number of input channels
    in_channels = 3
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
            'images_fp': images_fp,
            'lidar_fp': lidar_fp,
            'saved_model': saved_model,
            'in_channels': in_channels,
            'predictions': predictions,
            'n_classes': n_classes,
            'use_lidar': use_lidar
        }
    # Store the hyperparameters in a dictionary
    hyperparameters = {
                    'testing_batch_size': testing_batch_size,
                    'depth': depth,
                    'wf': wf,
                    'pad': pad,
                    'batch_norm': batch_norm,
                    'up_mode': up_mode
                }

    # Make sure the file paths exist
    if not os.path.isdir(images_fp):
        sys.exit('Error: Main file path to the test images does not exist')
    if not os.path.isdir(predictions):
        sys.exit('Error: Main file path to the predictions directory does not exist')
    if not os.path.isfile(os.path.join(os.getcwd(), 'saved_models', saved_model + '.pt')):
        sys.exit('Error: Saved model parameters file does not exist')
    if use_lidar and not os.path.isdir(lidar_fp):
        sys.exit('Error: Main file path to the lidar data does not exist.')

    return hyperparameters, options
