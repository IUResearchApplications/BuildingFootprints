# Building Footprints #
## File Structure ##
The structure of the file system for training and testing is set up like this:

* Name of Dataset
    * training_images
    * training_labels
    * validation_images
    * validation_labels
    * test_images
    * test_labels (Not required unless you want to run `metrics.py`)

You will need to specify where the dataset is located.

## Data Structure ##
The satellite images should be in RGB format. The labels need to be in greyscale format with pixel value 255 denoting a building and 0 denoting a non-building. The metadata needs to be stripped from both the satellite images and greyscale labels before training and testing. The recommended dimensions of the images and labels is 256x256. TIFF files are required if you want to polygonize them afterwards.

## Training and Testing ##
The UNet model used is from https://github.com/jvanvugt/pytorch-unet, which is based off of [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (Ronneberger et al., 2015). Refer to the links if you want to learn more about its implementation.

To use the model, start at the python program `setup.py`. In there you will specify the model parameters, where your datasets are, and where the predictions will be saved. If you want the images and labels to be padded beforehand rather than letting the model pad for you, you will need to use 256x256 images. If you have images of other dimensions, then you will need to enable padding from the model. Once that is set up run `train.py` to train the model. A directory called `saved_models` will be created that will store the model's learned parameters (i.e. weights and biases). This file can be quite large depending on the size of the model (up to 2gbs). Only the best parameters in the entire run will be saved. Once that is done run `test.py` to test the model. A directory called `predictions` will be created wherever you specified to save them. If you want to see the metrics from your test set run `metrics.py`, but you will need test labels in order to use it.

Currently, augmentation only consists of randomly flipping the images and labels vertically and horizontally. The training dataset will be doubled in size afterwards. If you want to increase the size of the dataset more, or apply different transformations, the augmentation code is in the function `train_data` in `load_dataset.py` located in directory `scripts`.

If you do not want to replace the predictions from a previous run, or have a new dataset, then I would recommend a couple of options. You could specify a different file path to save the predictions to or create a directory that will store new predictions from each run under a different file name (i.e. mainfilepath/runs/run1, mainfilepath/runs/run2, etc.).

## Polygonization ##
Refer to `README.md` in the `poly` directory.
