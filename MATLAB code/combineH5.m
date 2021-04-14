clear;
clc;

% This script is made to combine generated tilebox in order to create
% training and validation sets.


fileName = 'B40_P1024_TrainSet_CLS.h5';
inputFolder = '..\generatedData\';
outputFolder = '..\generatedData\trainingSet_SM_2\B40_P1024\';

maximumFileSize = 100;

proportion = 0.50;

mixH5Files(fileName,inputFolder,outputFolder,maximumFileSize,proportion,[]);%"segmentation")