# Building Footprints #
## Info ##
The UNet model used is from https://github.com/jvanvugt/pytorch-unet, which is based off of [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (Ronneberger et al., 2015). Refer to the links if you want to learn more about its implementation.

Currently, augmentation only consists of randomly flipping the images and labels vertically and horizontally. If you want to apply different transformations, the augmentation code is in `dataset_class.py` located in directory `scripts`. If you are including LiDAR data you cannot use this option to augment the data. You will need to augment the data beforehand.

If you do not want to replace the predictions from a previous run, or have a new dataset, then I would recommend a couple of options. You could specify a different file path to save the predictions to or create a directory that will store new predictions from each run under a different file name (i.e. mainfilepath/runs/run1, mainfilepath/runs/run2, etc.).

## Data Structure ##
The satellite images should be in RGB format. The labels need to be in greyscale format with pixel value 255 denoting a building and 0 denoting a non-building. The metadata needs to be stripped from both the satellite images and greyscale labels before training and testing. The recommended dimensions of the images and labels is 256x256. TIFF files are required to train, test, and polygonize.

The LiDAR data needs to be in text files with comma separated values. The model performs best if there is a height value to go with each pixel in the satellite images (No 0s). The min-max normalize values will need to be changed in `dataset_class.py` based on your LiDAR data.

## Saved Model Parameters ##
This model was trained on a sliced up (256x256 slices) satellite image of Hamilton County, with a grayscale image corresponding to the satellite image indicating where buildings are and aren't. The resolution of the imagery was 1ft. The model's saved parameters can be downloaded [here](https://drive.google.com/file/d/1uV8wF0_3cZFnhB8ZFkI20uRPjiGTBoVO/view?usp=sharing). LiDAR was also used to supplement the dataset and the saved model parameters from including LiDAR data can be found [here](https://drive.google.com/file/d/1R9ma3H3a8C5oJHYobgfaXMWOilNVczZ8/view?usp=sharing).

## File Structure For Training ##
The structure of the file system for training is set up like this:

* Name of Dataset
    * training_images
    * training_labels
    * validation_images
    * validation_labels
    * lidar (directory only needed if using LiDAR data)
        * lidar_training_data
        * lidar_validation_data

You will need to specify in `train_setup.py` where the dataset is located.

## Training ##
To train the model you will need to edit `train_setup.py`. In there you will specify the model parameters, where your dataset is, and where the predictions will be saved. If you want the images and labels to be padded beforehand rather than letting the model pad for you, you will need to use 256x256 images. If you have images of other dimensions, then you will need to enable padding from the model. If you are including LiDAR data you cannot have the images and labels padded before hand. Once that is set up run `train.py` to train the model. A directory called `saved_models` will be created that will store the model's learned parameters (i.e. weights and biases). This file can be quite large depending on the size of the model (up to 2gbs). The best parameters in the entire run will be saved and so will the parameters from the last completed epoch.

## Testing ##
To test the model you will need to edit `test_setup.py`. In there you need to give the file path to where the test images are located. If you are using LiDAR data then you will need to provide the file path to that as well. The saved learned parameters that are being used should be located within the directory `saved_models`. You will also need to give a file path to where you want the predictions to be saved. Once it is all set up you can run `test.py`.

## Metrics ##
If you want metrics to be calculated on your test dataset then you can use `metrics.py` to calculate the precision, recall, F1 score, and intersection over union (IoU). In `metrics_setup.py` you need to give the file path to the test dataset's labels and predictions. Once it is set up you can run `metrics.py`. It should be noted that the metrics are only calculated on predictions that have labels with buildings in them.

## Polygonization ##
Refer to `README.md` in the `poly` directory.
