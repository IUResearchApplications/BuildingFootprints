%%Code adapted from Mathworks segnetLayers help page
% load simple_net_eval 
% will load all the variables.
%%Define imagedatastore

imageDir = 'training_images';
labelDir = 'training_labels';

imds = imageDatastore(imageDir);

classNames = ["Building","Ground"];
labelIDs   = [255 0];
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

%%
imageSize = [255 255 3];
numClasses = 2;
lgraph = segnetLayers(imageSize,numClasses,2)

pximds = pixelLabelImageDatastore(imds,pxds);
options = trainingOptions('sgdm','InitialLearnRate',1e-3, ...
      'MaxEpochs',20,'VerboseFrequency',10);
%THIS NEXT STEP WILL TAKE HOURS TO RUN. LOAD THE FILE.
% tic
%  net = trainNetwork(pximds,lgraph,options)
% toc
 %save simple_net.mat
%%
%% 
% Define the location of the test images.
dataSetDir = '/N/dc2/projects/levee_ml/hamiltondemo/small_sample';
testImagesDir = fullfile(dataSetDir,'test_images');
%% 
% Create an |imageDatastore| object holding the test images.

imdsTest = imageDatastore(testImagesDir);
%% 
% Define the location of the ground truth labels.

testLabelsDir = fullfile(dataSetDir,'test_labels');
%% 
% Define the class names and their associated label IDs. The label IDs are the 
% pixel values used in the image files to represent each class.

classNames = ["Building" "Ground"];
labelIDs = [255 0];
%% 
% Create a |pixelLabelDatastore| object holding the ground truth pixel labels 
% for the test images.

pxdsTruth = pixelLabelDatastore(testLabelsDir,classNames,labelIDs);

%% 
% Run the network on the test images. Predicted labels are written to disk in 
% a temporary directory and returned as a |pixelLabelDatastore| object.

pxdsResults = semanticseg(imdsTest,net,"WriteLocation",tempdir);
%% Evaluate the Quality of the Prediction
% The predicted labels are compared to the ground truth labels. While the semantic 
% segmentation metrics are being computed, progress is printed to the Command 
% Window.

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);
%% Inspect Class Metrics
% Display the classification accuracy, the intersection over union (IoU), and 
% the boundary F-1 score for each class in the data set.

metrics.ClassMetrics
%% Display the Confusion Matrix
% Display the confusion matrix.

metrics.ConfusionMatrix
%% 
% Visualize the normalized confusion matrix as a heat map in a figure window.

normConfMatData = metrics.NormalizedConfusionMatrix.Variables;
figure
h = heatmap(classNames,classNames,100*normConfMatData);
h.XLabel = 'Predicted Class';
h.YLabel = 'True Class';
h.Title = 'Normalized Confusion Matrix (%)';
%% Inspect an Image Metric
% Visualize the histogram of the per-image intersection over union (IoU).

imageIoU = metrics.ImageMetrics.MeanIoU;
figure
histogram(imageIoU)
title('Image Mean IoU')
%% 
% Find the test image with the lowest IoU.

[minIoU, worstImageIndex] = min(imageIoU);
minIoU = minIoU(1);
worstImageIndex = worstImageIndex(1);
%% 
% Read the test image with the worst IoU, its ground truth labels, and its predicted 
% labels for comparison.

worstTestImage = readimage(imdsTest,worstImageIndex);
worstTrueLabels = readimage(pxdsTruth,worstImageIndex);
worstPredictedLabels = readimage(pxdsResults,worstImageIndex);
%% 
% Convert the label images to images that can be displayed in a figure window.

worstTrueLabelImage = im2uint8(worstTrueLabels == classNames(1));
worstPredictedLabelImage = im2uint8(worstPredictedLabels == classNames(1));
%% 
% Display the worst test image, the ground truth, and the prediction.

figure
montage({worstTestImage, worstTrueLabelImage, worstPredictedLabelImage},'Size',[1 3])
title(['Test Image vs. Truth vs. Prediction. IoU = ' num2str(minIoU)])
%% 
% Similarly, find the test image with the highest IoU.

[maxIoU, bestImageIndex] = max(imageIoU);
maxIoU = maxIoU(1);
bestImageIndex = bestImageIndex(1);
%% 
% Repeat the previous steps to read, convert, and display the test image with 
% the best IoU with its ground truth and predicted labels.

bestTestImage = readimage(imdsTest,bestImageIndex);
bestTrueLabels = readimage(pxdsTruth,bestImageIndex);
bestPredictedLabels = readimage(pxdsResults,bestImageIndex);

bestTrueLabelImage = im2uint8(bestTrueLabels == classNames(1));
bestPredictedLabelImage = (bestPredictedLabels == classNames(1));

figure
montage({bestTestImage, bestTrueLabelImage, bestPredictedLabelImage},'Size',[1 3])
title(['Test Image vs. Truth vs. Prediction. IoU = ' num2str(maxIoU)])
%% Specify Metrics to Evaluate
% Optionally, list the metric(s) you would like to evaluate using the |'Metrics'| 
% parameter.
% 
% Define the metrics to compute.

evaluationMetrics = ["accuracy" "iou"];
%% 
% Compute these metrics for the test data set.

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth,"Metrics",evaluationMetrics);
%% 
% Display the chosen metrics for each class.

metrics.ClassMetrics
%% 
save simple_net_eval.mat