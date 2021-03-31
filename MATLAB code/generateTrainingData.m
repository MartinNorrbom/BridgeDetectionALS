clear;
clc;

% This script is made to generate training data from LAZ files, the output
% is files in h5 format that contains tile blocks over the selected areas.

% Parameters for tile-blocks
gridSize             = [ 20, 30,  30,  40,  40,  50,  50,  60];
tileBlockPointNumber = [512,512,1024,1024,2048,2048,4096,4096];

% Info of LAZ file
% "class" represent the class that will be labeled in the training data.
% In this case 17 is bridges.
class = 17;
% Indicates the side length of the LAZ files.
% In this case it is 2.5 kilometers.
sizeIndexBlock = 2500;

% User parameters
gui = 1;
sepFiles = 1;

% All the paths that are needed.
CoordinatesPath = '..\selectedCoordinates\';
H5FileName = 'DebugTest1.h5';
generationFolder = '..\generatedData\';
serverName = "download-opendata.lantmateriet.se";
path1Server = '/Laserdata_Skog';
%pathData = "/Laserdata_Skog/2019/19B030/"; % Example path to file in server.
dataLAZPath = '..\dataFromLantmateriet\LAZdata\';
dataInfoPath = '..\dataFromLantmateriet\utfall_laserdata_skog_shape\';
laserDataLocationInfo = 'utfall_laserdata_skog.shp';

% Full file location for production status.
statusFile = [dataInfoPath,laserDataLocationInfo];

% Check if graphic interface to select file is active.
if(gui)
    % Filter for gui
    selectFilter = strcat(CoordinatesPath,"*.txt");
    % Run gui
    [file,path] = uigetfile( selectFilter,'MultiSelect','on');
    
    % Save file as text cell.
    if(~iscell(file))
        file = {file};
    end
    
    % Merge the entire location of the selected files.
    coordFiles = strcat(path,string(file));
    
else
    % Just take the files that is located in the folder.
    coordFiles = CoordinatesPath;
end

% Loop through all the parameter settings for tile blocks.
for ii=1:length(gridSize)
    % Check if the file should be saved separative.
    if(sepFiles && isstring(coordFiles))
        % Loop through each coordinate text file.
        for jj=1:length(coordFiles)
            % Define a name for the output file.
            nameCoord = erase(file(jj),["Koordinater",".txt"]);
            tempH5Name = strcat("B",string(gridSize(ii)),"_P",string(tileBlockPointNumber(ii)), ...
                "_G",nameCoord,".h5");
        
            % Generate the data.
            getTrainingData(gridSize(ii),tileBlockPointNumber(ii),class,sizeIndexBlock, ...
                tempH5Name,generationFolder,coordFiles(jj),...
                statusFile,dataLAZPath, ...
                serverName,path1Server)
            
        end
    else
        % Add the parameters to the file name.
        tempH5Name = strcat("B",string(gridSize(ii)),"_P",string(tileBlockPointNumber(ii)), ...
            "_",H5FileName);
        
        % Generate the training data.
        getTrainingData(gridSize(ii),tileBlockPointNumber(ii),class,sizeIndexBlock, ...
            tempH5Name,generationFolder,coordFiles, ...
            statusFile,dataLAZPath, ...
            serverName,path1Server)
    end
    
end




