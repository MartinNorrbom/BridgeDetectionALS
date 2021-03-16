clear;
clc;

% This script is made to combine generated tilebox in order to create
% training and validation sets.


fileName = 'debugTest.h5';
inputFolder = '..\generatedData\';
outputFolder = '..\generatedData\TestSet\';

maximumFileSize = 100;


mixH5Files(fileName,inputFolder,outputFolder,maximumFileSize)