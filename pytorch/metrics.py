import numpy as np
import os
import torch
import glob
import sys
from metrics_setup import setup_parameters
from PIL import Image

def recall_calc(true_1, false_0):
    rec = true_1 / float(true_1 + false_0)
    return rec

def precision_calc(true_1, false_1):
    prec = true_1 / float(true_1 + false_1)
    return prec

def IoU_calc(true_1, false_1, false_0):
    IoUnion = true_1 / float(false_1 + false_0 + true_1)
    return IoUnion

def f1_calc(precision, recall):
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score

def confusion_matrix(pred_segm, lb_segm, dim):
    # Set uint8 -> uint16 so we can have values larger than 255
    # merge the two images together with new values where:
    # 0 = correct non-building pixel guess
    # 1 = incorrect building pixel guess
    # 256 = incorrect non-building pixel guess
    # 257 = correct building pixel guess
    merged_maps = np.bitwise_or(np.left_shift(lb_segm.astype('uint16'), 8),
                                pred_segm.astype('uint16'))

    # count the amount for each bin ranging from 0-257 (only bins 0,1,256,257 matter)
    hist = np.bincount(np.ravel(merged_maps))

    # grab the non-zero bins
    nonzero = np.nonzero(hist)[0]

    pred, label = np.bitwise_and(255, nonzero), np.right_shift(nonzero, 8)

    # find the amount of classes there are
    class_count = np.array([pred, label]).max() + 1

    # create a square matrix that can support higher numbers 
    conf_matrix = np.zeros([class_count, class_count], dtype='uint64')

    # replace the zeros with the values found for each bin
    # in this case 0 = nonbuilding pixel, 1 = building pixel
    #         label
    #       0       1
    # 0 |true_0 |false_0|  prediction
    # 1 |false_1|true_1 |
    conf_matrix.put(pred * class_count + label, hist[nonzero])

    # calculate the metrics
    if conf_matrix[1][1] != 0:
        recall = recall_calc(conf_matrix[1][1], conf_matrix[0][1])
        precision = precision_calc(conf_matrix[1][1], conf_matrix[1][0])
        IoU = IoU_calc(conf_matrix[1][1], conf_matrix[1][0], conf_matrix[0][1])
    else:
        recall = 0
        precision = 0
        IoU = 0

    return recall, precision, IoU

def main(predictions_fp, labels_fp):
    # grab the file path to all of the images
    predictions = os.path.join(predictions_fp, '*.tif')
    labels = os.path.join(labels_fp, '*.tif')

    # create a list containing the file paths to the tif images
    pred_list = glob.glob(predictions)
    lb_list = glob.glob(labels)

    # sort the lists so that they can be compared with each other
    pred_list.sort()
    lb_list.sort()

    # grab the dimensions of the predictions
    image_PIL = Image.open(pred_list[0])
    dim = np.asarray(image_PIL).shape[1]
    image_PIL.close()

    num_masks = 0

    tot_recall = 0
    tot_precision = 0
    tot_IoU = 0

    print ('Calculating metrics...')
    for i in range(len(pred_list)):
        # load in the predictions and the labels (make sure the values are either 0 or 1)
        pred_PIL = Image.open(pred_list[i])
        pred_segm = np.asarray(pred_PIL) / 255
    
        label_PIL = Image.open(lb_list[i])
        lb_segm = np.asarray(label_PIL.resize((dim, dim))) / 255

        # set up the confusion matrix and calculate the metrics
        # make sure there are objects in the labels being compared to
        if np.sum(lb_segm) != 0:
            num_masks += 1
            recall, precision, IoU = confusion_matrix(pred_segm, lb_segm, dim)
            tot_recall += recall
            tot_precision += precision
            tot_IoU += IoU

        # close the PIL images
        pred_PIL.close()
        label_PIL.close()
        
    print ("Recall: " + str(tot_recall / num_masks))
    print ("Precision: " + str(tot_precision / num_masks))
    print ("IoU: " + str(tot_IoU / num_masks))
    print ("F1 score: " + str(f1_calc(tot_precision , tot_recall) / num_masks))

if __name__=='__main__':

    # grab the file paths
    predictions_fp, labels_fp = setup_parameters()

    main(predictions_fp, labels_fp)


