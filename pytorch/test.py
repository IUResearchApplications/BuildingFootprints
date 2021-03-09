import torch
import numpy as np
import glob
import os
import cv2
import sys
from scripts import dataset_class
from test_setup import setup_parameters
from scripts.load_dataset import test_data
from torchvision import transforms
from models import unet
from torch.utils.data import DataLoader

def crop(batch, orig_dim):
    # set up a torch tensor that is the same size as the batch to save to
    b = torch.zeros((batch.shape[0], orig_dim, orig_dim))
    crop_img = transforms.Compose([transforms.ToPILImage(),
                                  transforms.CenterCrop(orig_dim),
                                  transforms.ToTensor()])

    # crop each image back to its original dimension
    for i in range(batch.shape[0]):
        b[i] = crop_img(batch[i].cpu())

    return b


def save_pred(predictions, num_img, batch_size, output, images_fp, index):
    # grab the file paths to all of the test images
    fp_list = glob.glob(os.path.join(images_fp, '*.tif'))
    fp_list.sort()

    full_path = os.path.join(output, 'predictions')

    if not os.path.isdir(full_path):
        os.mkdir(full_path)
        print ("Created folder 'predictions'", flush=True)
    
    # loop through each batch to save the predictions
    for i in range(predictions.shape[0]):
        # grab the name of the tif file in the folder
        bs_name = os.path.basename(fp_list[i + index])

        # save the prediction as pred_tif_file_name
        # can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory
        # first.
        cv2.imwrite(os.path.join(full_path, 'pred_' + bs_name),
                    np.asarray(predictions[i,:,:].cpu() * 255))
        
def load_model(model, saved_model):
    # load the model to use GPU if possible
    cwd = os.getcwd()

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(os.path.join(cwd, 'saved_models', saved_model)))
    else:
        model.load_state_dict(torch.load(os.path.join(cwd, 'saved_models', saved_model),
                                                      map_location=torch.device('cpu')))
    return model

def best_pred(model, saved_model, pred_output, images_fp, batch_size, pad_size, orig_dim,
              num_test, test_loader):
    # load the model parameters that made the best test predictions
    model = load_model(model, saved_model + '.pt')

    # set the model to be ready to test
    model.eval()

    test_acc = 0.0
    index = 0

    # test the model
    print ("Testing model...")

    # run through the batches of test images
    with torch.no_grad():
        for test_imgs in test_loader:
            # use gpu for images and labels if possible
            if torch.cuda.is_available():
                test_imgs = test_imgs.cuda()

            # predict the classes           
            outputs = model(test_imgs)

            # find the prediction for each pixel of an image
            _, prediction = torch.max(outputs.data, 1)

            # if the image and label were padded beforehand crop it back into its original dimensions
            if pad_size != 0:
                prediction = crop(prediction.float(), orig_dim)

            save_pred(prediction.long(), num_test, batch_size, pred_output, images_fp, index)

            index += batch_size

    print ("Done.")

def main(hyperparameters, options):
    # grab the hyperparameters and options for training
    images_fp = options['images_fp']
    lidar_fp = options['lidar_fp']
    in_channels = options['in_channels'] 
    n_classes = options['n_classes']
    pred_output = options['predictions']
    use_lidar = options['use_lidar']
    saved_model = options['saved_model']
    batch_size = hyperparameters['testing_batch_size']
    depth = hyperparameters['depth']
    wf = hyperparameters['wf']
    padding = hyperparameters['pad']
    batch_norm = hyperparameters['batch_norm']
    up_mode = hyperparameters['up_mode']

    # use the UNet model in models dir
    # https://github.com/jvanvugt/pytorch-unet
    model = unet.UNet(in_channels = in_channels, n_classes = n_classes, depth = depth, wf = wf,
                      padding = padding, batch_norm = batch_norm, up_mode = up_mode)

    # load in the test dataset
    test_img, test_lidar_data, orig_dim, num_test = test_data(images_fp, lidar_fp, use_lidar)

    # set up the custom test class
    custom_test_class = dataset_class.CustomDatasetFromTif(test_img, [], test_lidar_data, num_test,
                                                           orig_dim, depth, test_set = True, 
                                                           use_lidar = use_lidar)

    # set up the test data loader
    test_loader = DataLoader(dataset = custom_test_class, batch_size = batch_size)
    
    pad_size = custom_test_class.pad_size

    # use GPU if available, https://pytorch.org/docs/stable/notes/cuda.html
    if torch.cuda.is_available():
        model = model.cuda()

    best_pred(model, saved_model, pred_output, images_fp, batch_size, pad_size, orig_dim,
              num_test, test_loader)

if __name__ == '__main__':

    # grab the file paths and parameters
    hyperparameters, options = setup_parameters()

    main(hyperparameters, options)  
