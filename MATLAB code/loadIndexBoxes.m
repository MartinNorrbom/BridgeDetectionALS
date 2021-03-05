clear;
clc;

% All the paths that are need.
CoordinatesPath = '..\selectedCoordinates\';
generationFolder = '..\generatedData';
CoordinatesFileName = 'Koordinater.txt';
serverName = "download-opendata.lantmateriet.se";
path1Server = '/Laserdata_Skog';
%pathData = "/Laserdata_Skog/2019/19B030/"; % Example path to file in server.
dataLAZPath = '..\dataFromLantmateriet\LAZdata\';
dataInfoPath = '..\dataFromLantmateriet\utfall_laserdata_skog_shape\';
laserDataPruductionStatus = 'utfall_laserdata_skog.dbf';
laserDataLocationInfo = 'utfall_laserdata_skog.shp';

%%
% Parameters for tile-blocks
gridSize = 50;
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
[coordBlock,intensityBlock,returnNumberBlock,pointLabel,blockLabel] = ... 
    generateTileBlocks(fileForCoordinates,generationMethod,gridSize, ...
    tileBlockPointNumber,class,"dataPath",dataLAZPath,"neighbours",neighbourFiles);

%%
% ---------------- Plot all the generate tile-blocks ----------------

numberOfBlocks = size(coordBlock,3);
% for ii=1:numberOfBlocks
%     pcshow(coordBlock(:,:,ii)', intensityPlot(intensityBlock(1,:,ii),6))
%     w = waitforbuttonpress;
% end

% --- Make clusters to make better visualization of the data. ---
% GMModel = fitgmdist(dataSetSkog(:,:,jj)',5);
% idx = cluster(GMModel,dataSetSkog(:,:,jj)');

%%

normalEmpty = single(zeros([1 tileBlockPointNumber numberOfBlocks]));
pointFeatures = cat(1,intensityBlock,returnNumberBlock,normalEmpty);

dataNum = 1:tileBlockPointNumber;

% Create .h5 Data format
delete('trainingDataSkog.h5');
h5create('trainingDataSkog.h5','/data',[3 tileBlockPointNumber numberOfBlocks],'Chunksize',[1 128 103], ...
    'Deflate',4,'Datatype', 'single')
h5create('trainingDataSkog.h5','/label',[1 numberOfBlocks],'Chunksize',[1 numberOfBlocks], 'Deflate',1,'Datatype','int8');
h5create('trainingDataSkog.h5','/pid',[tileBlockPointNumber numberOfBlocks],'Chunksize',[256 128], 'Deflate',1,'Datatype','int8');
h5create('trainingDataSkog.h5','/normal',[3 tileBlockPointNumber numberOfBlocks],'Chunksize',[1 128 103], 'Deflate',4,'Datatype','single');


% Write the tile blocks data to h5 format
h5write('trainingDataSkog.h5','/data',coordBlock );
h5write('trainingDataSkog.h5','/label',blockLabel );
h5write('trainingDataSkog.h5','/pid',pointLabel );
h5write('trainingDataSkog.h5','/normal',pointFeatures );

%h5write('trainingDataSkog.h5','/data_num',dataNum );

%%
dataRead = h5read('trainingDataSkog.h5','/data');
blockLabelRead = h5read('trainingDataSkog.h5','/label');
pointLabelRead = h5read('trainingDataSkog.h5','/pid');
pointFeatureRead = h5read('trainingDataSkog.h5','/normal');



