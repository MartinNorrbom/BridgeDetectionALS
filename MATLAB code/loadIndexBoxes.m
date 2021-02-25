clear;
clc;

% All the paths that are need.
CoordinatesPath = '..\selectedCoordinates\';
CoordinatesFileName = 'Koordinater.txt';
pathWeb = "download-opendata.lantmateriet.se";
path1Server = '/Laserdata_Skog';
%pathData = "/Laserdata_Skog/2019/19B030/"; % Example path
dataLAZPath = '..\dataFromLantmateriet\LAZdata\';
dataInfoPath = '..\dataFromLantmateriet\utfall_laserdata_skog_shape\';
laserDataPruductionStatus = 'utfall_laserdata_skog.dbf';
laserDataLocationInfo = 'utfall_laserdata_skog.shp';


%%
% Read the data for the selected coordinates and production info of the data 
% provided by lantmäteriet.
fileID = fopen([CoordinatesPath,CoordinatesFileName],'r');
textDataSelectedPoints = textscan(fileID,'%c');
fclose(fileID);
dataLocationInfo = shaperead([dataInfoPath,laserDataLocationInfo]);

% Get the "SWEREF 99 TM" coordinates
textData = cell2mat(textDataSelectedPoints)';
startRef = 'SWEREF99TM:N';
endRef = '*FörberäkningariExcelmm*';

% Get the "SWEREF 99 TM" coordinates from the text file downloaded from the website:
% https://karthavet.havochvatten.se/visakoordinater/
startRefIndex = strfind(textData,startRef)+length(startRef);
endRefIndex = strfind(textData,endRef)-1;
importantText = textData(startRefIndex:endRefIndex);
unstructedCoords = split(importantText,["N","E"]);
manualSelectedCoordinates = {unstructedCoords(1:2:end),unstructedCoords(2:2:end)};

%%
% Find the LAZ-files for the selected coordinates.
[filePathsAndNames,fileForCoordinates] = getLAZFileFromCoord(manualSelectedCoordinates, dataLocationInfo);

%%
% Get all the file names for the LAZ files located in the data folder.
dataFolderLAZ = dir(fullfile(dataLAZPath,'*.laz'));
allFileNames = cell2mat({dataFolderLAZ.name});

if(isempty(allFileNames))
    allFileNames = '';
end

noCatch = true;
cFolder = path1Server;
ftpobj = [];

% ------------- Download all the missing files ----------------

% Loop through all the regions.
for ii=1:size(filePathsAndNames{1},1)
    
    % Check if there are index boxes located in the current region.
    if( ~isempty( filePathsAndNames{1}{ii} ) )
        
        % If at least one file is missing and the current index box is 
        % missing, locate the path to the current index box in the server.
        if (noCatch == false)
            if ~contains(cFolder,[path1Server,filePathsAndNames{1}{ii}] )
                cFolder = [path1Server,filePathsAndNames{1}{ii}];
                cd(ftpobj,cFolder);
            end
        end
        
        lengthJJ = size(filePathsAndNames{2}{ii},1);
        for jj=1:lengthJJ

            % Download the LAZ file if it is not downloaded.
            if( ~contains(allFileNames,filePathsAndNames{2}{ii}(jj,:)) )
                
                % Check if an connection to the ftp server is establish.
                if(noCatch)
                    noCatch = false;
                    disp("At least one file is missing. Log in to download the files.")
                    
                    % Write user name and password for the ftp-server.
                    UserName = input('Write user name: \n','s');
                    Password = input('Write password: \n','s');
                    
                    % Connect and login to the ftp server
                    ftpobj = ftp(pathWeb,UserName,Password);
                    
                    % Get to the folder in the ftp server where the missing 
                    % index box is located.
                    cFolder = [path1Server,filePathsAndNames{1}{ii}];
                    cd(ftpobj,cFolder);
                end
                
                % Download missing index box
                disp(['Download: ',filePathsAndNames{2}{ii}(jj,:)])
                mget(ftpobj,filePathsAndNames{2}{ii}(jj,:),dataLAZPath);
                
            end
        end
    end
end

% Close ftp server connection.
if (noCatch == false)
    close(ftpobj);
end

%%

% ------ Generate tile-blocks for the selected coordinates ------

% Parameters for tile-blocks
gridSize = 100;
tileBlockPointNumber = 2048;
class = 17;

% Generate tile blocks for the location of the selected coordinates.
for ii=1:size(fileForCoordinates,1)
    
    % Get file name for index box and coordinates within the index block.
    coordinates = fileForCoordinates{ii,2};
    LAZfilename = fileForCoordinates{ii,1};
    selectedFile = dir(fullfile(dataLAZPath,['*',fileForCoordinates{ii,1}]));
    
    % Check if the current file is available.
    if( ~isempty(selectedFile) )
    
        % Upload LAZ-file and get classes and point features.
        lasReader = lasFileReader([dataLAZPath,selectedFile.name]);
        [ptCloud,pointAttributes] = readPointCloud(lasReader,'Attributes',["Classification","LaserReturns"]);

        % Return tile blocks of selected coordinates.
        [dataSetSkog,returnNumberBlock,intensityBlock,pointLabel,blockLabel] = ...
           getBlockFromCoord(ptCloud,pointAttributes,class,tileBlockPointNumber,gridSize, flip(coordinates,2));

       % Plot each tile-block
        for jj=1:size(dataSetSkog,3)
            pcshow(dataSetSkog(:,:,jj)', intensityPlot(intensityBlock(1,:,jj),3))
            w = waitforbuttonpress;
        end
    end
end


% --- Make clusters to get the data more detailed ---
% GMModel = fitgmdist(dataSetSkog(:,:,jj)',5);
% idx = cluster(GMModel,dataSetSkog(:,:,jj)');

% --- Commands for the ftp-server ---
% ftpobj = ftp(pathWeb,UserName,Password);
% cd(ftpobj,pathData);
% mget(ftpobj,dataName); % '19B030_65825_4150_25.laz'
% close(ftpobj);