clear;
clc;

% All the paths that are need.
CoordinatesPath = '..\selectedCoordinates\';
H5FileName = 'B30_1024_Skane_Akermark_pointcnn.h5';
generationFolder = '..\generatedData\';
serverName = "download-opendata.lantmateriet.se";
path1Server = '/Laserdata_Skog';
%pathData = "/Laserdata_Skog/2019/19B030/"; % Example path to file in server.
dataLAZPath = '..\dataFromLantmateriet\LAZdata\';
dataInfoPath = '..\dataFromLantmateriet\utfall_laserdata_skog_shape\';
laserDataPruductionStatus = 'utfall_laserdata_skog.dbf';
laserDataLocationInfo = 'utfall_laserdata_skog.shp';

%%
% Parameters for tile-blocks
gridSize = 30;
tileBlockPointNumber = 1024;
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
[coordBlock,intensityBlock,returnNumberBlock,pointLabel,blockLabel,blockGeoCoord] = ... 
    generateTileBlocks(fileForCoordinates,generationMethod,gridSize, ...
    tileBlockPointNumber,class,"dataPath",dataLAZPath,"neighbours",neighbourFiles);

%%
% ---------------- Plot all the generate tile-blocks ----------------

numberOfGeneratedBlocks = size(coordBlock,3);
% for ii=1:numberOfGeneratedBlocks
%     pcshow(coordBlock(:,:,ii)', intensityPlot(intensityBlock(1,:,ii),6))
%     w = waitforbuttonpress;
% end

% --- Make clusters to make better visualization of the data. ---
% GMModel = fitgmdist(dataSetSkog(:,:,jj)',5);
% idx = cluster(GMModel,dataSetSkog(:,:,jj)');

%%
% To save a balanced training set.

% indexNonB = find(blockLabel == 0);
% indexB = find(blockLabel == 1);
% 
% randNonB = randperm(length(indexNonB),length(indexB));
% 
% indToSave = sort([indexB,indexNonB(randNonB)]);

%%
numberOfBlocks = size(coordBlock,3); %length(indToSave);

dataNum = int32(1:numberOfBlocks);

% GOT TO FEW GEOGRAPHIC COORDINATES!!!
% Create .h5 Data format
saveTileBlocksH5(H5FileName,coordBlock,blockLabel,pointLabel, ...
    "data_num",dataNum,"intensity",intensityBlock, ...
    "returnNumber",returnNumberBlock,"path",generationFolder, ...
    "coordinates",blockGeoCoord');

% % Commands to read data.
% dataRead = h5read([generationFolder,H5FileName],'/data');
% blockLabelRead = h5read([generationFolder,H5FileName],'/label');
% pointLabelRead = h5read([generationFolder,H5FileName],'/pid');
% pointFeatureRead = h5read([generationFolder,H5FileName],'/normal');

% % Create h5 data format for pointNet
% delete('trainingDataSkog.h5');
% h5create('trainingDataSkog.h5','/data',[3 tileBlockPointNumber numberOfBlocks],'Chunksize',[1 128 103], ...
%     'Deflate',4,'Datatype', 'single')
% h5create('trainingDataSkog.h5','/label',[1 numberOfBlocks],'Chunksize',[1 numberOfBlocks], 'Deflate',1,'Datatype','int8');
% h5create('trainingDataSkog.h5','/pid',[tileBlockPointNumber numberOfBlocks],'Chunksize',[256 128], 'Deflate',1,'Datatype','int8');
% h5create('trainingDataSkog.h5','/normal',[3 tileBlockPointNumber numberOfBlocks],'Chunksize',[1 128 103], 'Deflate',4,'Datatype','single');
% h5create('trainingDataSkog.h5','/data_num',[1 numberOfBlocks],'Chunksize',[1 numberOfBlocks], 'Deflate',1,'Datatype','int32');
% h5create('trainingDataSkog.h5','/indices_split_to_full',[1 tileBlockPointNumber numberOfBlocks],'Chunksize',[1 128 103], 'Deflate',1,'Datatype','int32');

