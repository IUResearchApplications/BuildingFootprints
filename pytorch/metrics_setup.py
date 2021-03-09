import sys
import os

def setup_parameters():
    """
    This function sets up the file paths to the labels and predictions to compute the metrics on.
    
    Returns:
    predictions_fp: Main file path to the predictions.
    labels_fp: Main file path to the labels.
    """

    # Labels location. Need to be .tif files.
    labels_fp = "/file/path/to/labels"

    # Predictions location. Need to be .tif files.
    predictions_fp = "/file/path/to/predictions"

    # Make sure the file paths exist
    if not os.path.isdir(predictions_fp):
        sys.exit('Error: Main file path to the predictions does not exist')
    if not os.path.isdir(labels_fp):
        sys.exit('Error: Main file path to the labels does not exist')

    return predictions_fp, labels_fp

