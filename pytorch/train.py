import numpy as np
import torch
import os
import sys
from scripts import dataset_class
from setup import setup_parameters
from scripts.load_dataset import train_data
from metrics import confusion_matrix
from models import unet
from torch.utils.data import DataLoader
from torchvision import transforms

# https://github.com/jvanvugt/pytorch-unet , https://github.com/jeffwen/road_building_extraction
def save_model(model):
    # save the model's learned parameters at the current epoch to the saved_models dir
    # if the folder 'saved_models' doesn't exist then create one
    full_path = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.isdir(full_path):
        os.mkdir(full_path)
        print ("Created folder 'saved_models'", flush=True)

    save_fp = os.path.join(full_path, 'Best_UNetmodel.model')
    torch.save(model.state_dict(), save_fp)
    print ("Checkpoint saved", flush=True)

def adjust_lr(epoch, learn_rate, optimizer, lr_change):
    # adjust the learning rate after a certain amount of epochs so the model can perform better
    adjust = int(epoch / lr_change)
    
    learn_rate = learn_rate / (10 ** adjust)

    # update the learning rate in our optimizer
    for param_group in optimizer.param_groups:
        param_group["lr"] = learn_rate

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

def train(model, training_loader, optimizer, loss, pad_size, orig_dim, num_img):
    # set the model to be ready to train
    model.train()

    train_acc = 0.0
    train_loss = 0.0

    # train the model
    print ("Training model...", flush=True)  
    # run through the batches of training images and labels
    for train_imgs, train_labels in training_loader:
        
        # use GPU for images and labels if possible
        if torch.cuda.is_available():
            train_imgs = train_imgs.cuda()
            train_labels = train_labels.cuda()

        # clear accumulated gradients
        optimizer.zero_grad()
                
        # predict the classes
        outputs = model(train_imgs)

        # using cross entropy as loss function per https://arxiv.org/pdf/1505.04597.pdf
        # compute the loss
        loss_output = loss(outputs, train_labels.long())

        # backpropagate the loss
        loss_output.backward()

        # adjust parameters according to computed gradients
        optimizer.step()      

        # calculate the training loss and find the prediction for each pixel of an image
        train_loss += loss_output.item() * train_imgs.size(0)
        _, prediction = torch.max(outputs.data, 1)

        # if the image and label were padded beforehand crop it back into its original dimensions
        if pad_size !=0:
            prediction = crop(prediction.float(), orig_dim)
            train_labels = crop(train_labels, orig_dim)

        # calculate the accuracy after each batch
        train_acc += torch.sum(prediction.long() == train_labels.long().data)


    # calculate the total accuracy and training loss
    train_acc = train_acc.item() / (float(num_img) * orig_dim * orig_dim)
    train_loss = train_loss / num_img

    print ("Train acc: " + str(train_acc), "Train loss: " + str(train_loss), flush=True)  


def valid(model, validation_loader, loss, best_loss, pad_size, orig_dim, num_img):
    # set the model to be ready to test on the validation set
    model.eval()

    validation_acc = 0.0
    validation_loss = 0.0
    validation_IoU = 0.0
    IoU_count = 0.0

    # test the model
    print ("Testing model...", flush=True)

    # run through the batches of validation images
    for validation_imgs, validation_labels in validation_loader:
        # use GPU for images if possible
        if torch.cuda.is_available():
            validation_imgs = validation_imgs.cuda()
            validation_labels = validation_labels.cuda()

        # predict the classes (whether a pixel contains a building or not)            
        outputs = model(validation_imgs)

        # compute the loss
        loss_output = loss(outputs, validation_labels.long())

        # calculate the validation loss and find the prediction for each pixel of an image
        validation_loss += loss_output.item() * validation_imgs.size(0)
        _, prediction = torch.max(outputs.data, 1)

        # if the image and label were padded beforehand crop them back into their original
        # dimensions
        if pad_size != 0:
            prediction = crop(prediction.float(), orig_dim)
            validation_labels = crop(validation_labels, orig_dim)

        for img in range(validation_labels.shape[0]):
            # set up the confusion matrix and calculate the metrics
            # check that there is an object to compute the metrics with
            if torch.sum(validation_labels[img]) != 0:
                IoU_count += 1
                _,_,IoU = confusion_matrix(np.asarray(prediction[img].cpu()),
                                           np.asarray(validation_labels[img].cpu()), orig_dim)
                validation_IoU += IoU

        # calculate the accuracy after each batch
        validation_acc += torch.sum(prediction.long() == validation_labels.long().data)
            
    # calculate the total accuracy, validation loss, and IoU
    validation_acc = validation_acc.item() / (float(num_img) * orig_dim * orig_dim)
    validation_loss = validation_loss / num_img

    # make sure there is no divide by 0
    if IoU_count != 0:
        validation_IoU = validation_IoU / IoU_count

    print ("Val acc: " + str(validation_acc), "Val loss: " + str(validation_loss), flush=True)
    print ("Val IoU: " + str(validation_IoU), flush=True)
    
    # save the model's learned parameters with the lowest loss
    if best_loss > validation_loss:
        best_loss = validation_loss
        save_model(model)

    return best_loss

def main(hyperparameters, options):
    # grab the hyperparameters and options for training
    data_set =      options['dataset']  
    in_channels =   options['in_channels'] 
    n_classes =     options['n_classes'] 
    augment =       options['augment']
    class_weights =         hyperparameters['class_weights'] 
    num_epochs =            hyperparameters['epochs']
    learning_rate =         hyperparameters['learn_rate']
    lr_change =             hyperparameters['lr_change']
    training_batch_size =   hyperparameters['training_batch_size']
    testing_batch_size =    hyperparameters['testing_batch_size']
    depth =                 hyperparameters['depth']
    wf =                    hyperparameters['wf']
    padding =               hyperparameters['pad']
    batch_norm =            hyperparameters['batch_norm']
    up_mode =               hyperparameters['up_mode']

    print ("""Running model with epochs={}, learning_rate={}, training_batch_size={},
testing_batch_size={}, in_channels={}, n_classes={}, depth={}, wf={}, padding={},
batch_norm={}, up_Mode={}, augment={}""".format(num_epochs, learning_rate, training_batch_size,
                                                testing_batch_size, in_channels, n_classes,
                                                depth, wf, padding, batch_norm, up_mode,
                                                augment), flush=True)

    # use the UNet model in models dir
    # https://github.com/jvanvugt/pytorch-unet
    model = unet.UNet(in_channels = in_channels, n_classes = n_classes, depth = depth, wf = wf,
                      padding = padding, batch_norm = batch_norm, up_mode = up_mode)

    # use GPU if available, https://pytorch.org/docs/stable/notes/cuda.html
    if torch.cuda.is_available():
        model = model.cuda()

    # set up the optimizer for our model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

    # give more weight to building predictions
    loss_weights = torch.tensor(class_weights)
    # use GPU if available
    if torch.cuda.is_available():
        loss_weights = loss_weights.cuda()

    # set up the loss with the weights that were specified
    loss = torch.nn.CrossEntropyLoss(weight = loss_weights)    

    # starting epoch
    start_epoch = 1

    # best validation loss
    best_loss = 10000    

    # load in the training dataset
    train_img, train_label, orig_dim, num_train, pad_size = train_data(data_set, depth, padding,
                                                            augment, current_set = 'training')

    # set up the custom training class
    custom_training_class = dataset_class.CustomDatasetFromTif(train_img, train_label, num_train)

    # set up the training data loader
    training_loader = DataLoader(dataset=custom_training_class, batch_size=training_batch_size)

    valid_img, valid_label, orig_dim, num_valid, pad_size = train_data(data_set, depth, padding,
                                                            augment, current_set = 'validation')

    # set up the custom validation class
    custom_validation_class = dataset_class.CustomDatasetFromTif(valid_img, valid_label,
                                                                 num_valid)

    # set up the validation data loader
    validation_loader = DataLoader(dataset=custom_validation_class,
                                   batch_size=testing_batch_size)   

    # loop through all of the epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print("Epoch "+str(epoch), flush=True)

        # adjust the learning rate after a certain amount of epochs
        if epoch % lr_change == 0:
            adjust_lr(epoch, learning_rate, optimizer, lr_change)

        # run train and valid
        train(model, training_loader, optimizer, loss, pad_size, orig_dim, num_train)
        best_loss = valid(model, validation_loader, loss, best_loss, pad_size, orig_dim,
                          num_valid)

        # shuffle the training dataset after every epoch
        custom_training_class = dataset_class.CustomDatasetFromTif(train_img, train_label,
                                                                   num_train)

        training_loader = DataLoader(dataset=custom_training_class,
                                     batch_size=training_batch_size)

if __name__ == '__main__':

    # hyperparameters and options for the model
    hyperparameters, options = setup_parameters('training')

    main(hyperparameters, options)
