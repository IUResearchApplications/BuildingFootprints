import sys
import os

def setup_parameters(status):
    """
    This function sets up the hyperparameters needed to run train.py (for training), test.py (for
    testing), and metrics.py (for calculating the metrics for test.py). Hyperparameters given to
    test.py must match the hyperparameters used in train.py when the model's learned parameters
    were saved.

    Parameters:
    -----------
    status: Specifies which parameters to return
    
    Returns:
    hyperparameters : Dictionary containing the hyperparameters for the model
    options : Dictionary containing the options for the dataset location, augmentation, etc.
    """

    # Dataset location (for training, testing, and metrics)
    dataset = '/file/path/to/data/'

    ### FOR TRAINING ###

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

    ### FOR TRAINING + TESTING ###

    # Size of the batch fed into the model. Model can handle a larger batch size when training.
    # Batch size used during training
    training_batch_size = 10
    # Batch size used during testing
    testing_batch_size = 5

    ### FOR TESTING ###

    # Model's learned parameters (i.e. weights and biases) that achieved the lowest loss
    saved_model = 'best_unet_model.pt'

    ### FOR TESTING + METRICS ###

    # Main file path to where the predictions are saved (folder 'predictions' will be created)
    predictions = '/file/path/to/predictions/'

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
            'dataset': dataset,
            'saved_model': saved_model,
            'in_channels': in_channels,
            'predictions': predictions,
            'n_classes': n_classes,
            'augment': augment
        }
    # Store the hyperparameters in a dictionary
    hyperparameters = {
                    'epochs': epochs,
                    'learn_rate': learn_rate,
                    'lr_change': lr_change,
                    'class_weights': class_weights,
                    'training_batch_size': training_batch_size,
                    'testing_batch_size': testing_batch_size,
                    'in_channels': in_channels,
                    'depth': depth,
                    'wf': wf,
                    'pad': pad,
                    'batch_norm': batch_norm,
                    'up_mode': up_mode
                }

    if status == 'training':
        # Make sure the file path exist
        if not os.path.isdir(dataset):
            sys.exit('Error: Main file path to the training and validation images does not exist')

        return hyperparameters, options

    elif status == 'testing':
        # Make sure the file paths exist
        if not os.path.isdir(dataset):
            sys.exit('Error: Main file path to the test images does not exist')
        if not os.path.isdir(predictions):
            sys.exit('Error: Main file path to the predictions directory does not exist')
        if not os.path.isfile(os.path.join(os.getcwd(), 'saved_models', saved_model)):
            sys.exit('Error: Saved model parameters file does not exist')

        return hyperparameters, options

    else:
        labels = os.path.join(dataset, 'test_labels')
        predictions = os.path.join(predictions, 'predictions')

        # Make sure the file paths exist
        if not os.path.isdir(predictions):
            sys.exit('Error: Main file path to the predictions does not exist')
        if not os.path.isdir(labels):
            sys.exit('Error: Main file path to the test labels does not exist')

        return predictions, labels

