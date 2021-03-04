clear;
clc;

% All the paths that are need.
CoordinatesPath = '..\selectedCoordinates\';
CoordinatesFileName = 'Koordinater.txt';
serverName = "download-opendata.lantmateriet.se";
path1Server = '/Laserdata_Skog';
%pathData = "/Laserdata_Skog/2019/19B030/"; % Example path
dataLAZPath = '..\dataFromLantmateriet\LAZdata\';
dataInfoPath = '..\dataFromLantmateriet\utfall_laserdata_skog_shape\';
laserDataPruductionStatus = 'utfall_laserdata_skog.dbf';
laserDataLocationInfo = 'utfall_laserdata_skog.shp';

%%
% Parameters for tile-blocks
gridSize = 100;
tileBlockPointNumber = 20000;
class = 17;
sizeIndexBlock = 2500;

%%
%manualSelectedCoordinates = readCoordinates(CoordinatesPath,CoordinatesFileName,20);

[manualSelectedCoordinates,tileBlockMethod] = ...
    getSelectedCoordinates(CoordinatesPath,sizeIndexBlock,gridSize);

%%
dataLocationInfo = shaperead([dataInfoPath,laserDataLocationInfo]);
%%
% Find the LAZ-files for the selected coordinates.
[filePathsAndNames,fileForCoordinates,generationMethod,neighbourFiles] = ...
    getLAZFileFromCoord(manualSelectedCoordinates, dataLocationInfo,"Methods",tileBlockMethod,"neighbours");

%%
% ------------- Download all the missing files ----------------

getMissingFilesFromServer(filePathsAndNames,serverName,path1Server,dataLAZPath);
%%

% ------ Generate tile-blocks for the selected coordinates ------

% generationMethod = zeros(size(fileForCoordinates,1),3);
% generationMethod(:,[1,2,3]) = 1;
% Generate the tile blocks.
[coordBlock,intensityBlock,returnNumberBlock,pointLabel,blockLabel] = ... 
    generateTileBlocks(fileForCoordinates,generationMethod,gridSize, ...
    tileBlockPointNumber,class,"dataPath",dataLAZPath,"neighbours",neighbourFiles);

%%
% ---------------- Plot all the generate tile-blocks ----------------

for ii=1:size(coordBlock,3)
    pcshow(coordBlock(:,:,ii)', intensityPlot(intensityBlock(1,:,ii),6))
    w = waitforbuttonpress;
end

% --- Make clusters to make better visualization of the data. ---
% GMModel = fitgmdist(dataSetSkog(:,:,jj)',5);
% idx = cluster(GMModel,dataSetSkog(:,:,jj)');