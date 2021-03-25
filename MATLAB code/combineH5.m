clear;
clc;

% This script is made to combine generated tilebox in order to create
% training and validation sets.


fileName = 'B30_P1024_TrainSet_SEG.h5';
inputFolder = '..\generatedData\';
outputFolder = '..\generatedData\TrainingSet\';

maximumFileSize = 100;

proportion = 0.50;

mixH5Files(fileName,inputFolder,outputFolder,maximumFileSize,proportion,[]);%,"segmentation")