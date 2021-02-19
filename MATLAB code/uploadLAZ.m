clear;
clc;

% Load data
LAZfilename = "19B030_65825_4150_25.laz";

lasReader = lasFileReader(LAZfilename);
% "Classification","GPSTimeStamp","LaserReturns","NearIR","ScanAngle"
[ptCloud,pointAttributes] = readPointCloud(lasReader,'Attributes',["Classification","LaserReturns"]);

%%
% 
% % Create .h5 Data format
% delete('trainingDataSkog.h5');
% h5create('trainingDataSkog.h5','/data',[3 2048 1648],'Chunksize',[1 128 103], ...
%     'Deflate',4,'Datatype', 'single')
% h5create('trainingDataSkog.h5','/label',[1 1648],'Chunksize',[1 1648], 'Deflate',1,'Datatype','int8');
% h5create('trainingDataSkog.h5','/pid',[2048 1648],'Chunksize',[256 128], 'Deflate',1,'Datatype','int8');
% h5create('trainingDataSkog.h5','/normal',[3 2048 1648],'Chunksize',[1 128 103], 'Deflate',4,'Datatype','single');



% % Create a 3x2048x1648 empty matrix
% dataSetSkog = single(zeros([3 2048 1648]));
% returnNumberBlock = single(zeros([1 2048 1648]));
% intensityBlock = single(zeros([1 2048 1648]));


%%
% Create a tile block 
gridSize = 100; % 100m x 100m grid size
tileBlockPointNumber = 2048;
bridgeClassNumber = 17;

[dataBridge,intensityBridge,returnNumberBridge,pointLabelBridge,blockLabelBridge] = getBridgeBlock(ptCloud,pointAttributes,bridgeClassNumber,tileBlockPointNumber,gridSize);
[dataNonBridge,intensityNonBridge,returnNumberNonBridge,pointLabelNonBridge,blockLabelNonBridge] = getNonbridgeBlock(ptCloud,pointAttributes,bridgeClassNumber,tileBlockPointNumber,gridSize);
%%

numberOfBlocks = size(dataBridge,3)+size(dataNonBridge,3);

XYZdata = cat(3,dataBridge,dataNonBridge);
blockLabelData = cat(2,blockLabelBridge,blockLabelNonBridge);
pointLabelData = cat(2,pointLabelBridge,pointLabelNonBridge);


% Merge extra point feature
intensityData = cat(3,intensityBridge,intensityNonBridge);
returnNumberData = cat(3,returnNumberBridge,returnNumberNonBridge);
normalEmpty = single(zeros([1 tileBlockPointNumber numberOfBlocks]));

pointFeatureData = cat(1,intensityData,returnNumberData,normalEmpty);

%%
% Create .h5 Data format
delete('trainingDataSkog.h5');
h5create('trainingDataSkog.h5','/data',[3 tileBlockPointNumber numberOfBlocks],'Chunksize',[1 128 103], ...
    'Deflate',4,'Datatype', 'single')
h5create('trainingDataSkog.h5','/label',[1 numberOfBlocks],'Chunksize',[1 numberOfBlocks], 'Deflate',1,'Datatype','int8');
h5create('trainingDataSkog.h5','/pid',[tileBlockPointNumber numberOfBlocks],'Chunksize',[256 128], 'Deflate',1,'Datatype','int8');
h5create('trainingDataSkog.h5','/normal',[3 tileBlockPointNumber numberOfBlocks],'Chunksize',[1 128 103], 'Deflate',4,'Datatype','single');

% Write the tile blocks data to h5 format
h5write('trainingDataSkog.h5','/data',XYZdata );
h5write('trainingDataSkog.h5','/label',blockLabelData );
h5write('trainingDataSkog.h5','/pid',pointLabelData );
h5write('trainingDataSkog.h5','/normal',pointFeatureData );

%%
dataRead = h5read('trainingDataSkog.h5','/data');
blockLabelRead = h5read('trainingDataSkog.h5','/label');
pointLabelRead = h5read('trainingDataSkog.h5','/pid');
pointFeatureRead = h5read('trainingDataSkog.h5','/normal');



% %pcshow(dataRead(:,:,10)', returnNumberBlock(1,:,10))
% pcshow(dataRead(:,:,10)', intensityPlot(intensityBlock(1,:,10),10));


