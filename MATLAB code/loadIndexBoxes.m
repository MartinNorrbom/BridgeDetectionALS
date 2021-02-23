clear;
clc;

CoordinatesFileName = 'Koordinater.txt';
pathWeb = "download-opendata.lantmateriet.se";
path1Server = '/Laserdata_Skog';
%pathData = "/Laserdata_Skog/2019/19B030/"; % Example path
dataLAZPath = '..\LantmäterietsData\LAZdata\';
dataInfoPath = '..\LantmäterietsData\utfall_laserdata_skog_shape\';
laserDataPruductionStatus = 'utfall_laserdata_skog.dbf';
laserDataLocationInfo = 'utfall_laserdata_skog.shp';


%%
% Read the data.
fileID = fopen(CoordinatesFileName,'r');
textDataSelectedPoints = textscan(fileID,'%c');

dataLocationInfo = shaperead([dataInfoPath,laserDataLocationInfo]);

% Get the SWEREF 99 TM coordinates
textData = cell2mat(textDataSelectedPoints)';
startRef = 'SWEREF99TM:N';
endRef = '*FörberäkningariExcelmm*';

startRefIndex = strfind(textData,startRef)+length(startRef);
endRefIndex = strfind(textData,endRef)-1;

importantText = textData(startRefIndex:endRefIndex);

unstructedCoords = split(importantText,["N","E"]);
manualSelectedCoordinates = {unstructedCoords(1:2:end),unstructedCoords(2:2:end)};

%%
% Find LAZ-file for the coordinates
% Example coordinate
%exCoord = {{'6583806';'6673806';'6583492';'7308464'},{'415537';'440037';'586142';'818822'}};

[filePathsAndNames,fileForCoordinates] = getLAZFileFromCoord(manualSelectedCoordinates, dataLocationInfo);

%%
% Get file names in the laser data folder.
dataFolderLAZ = dir(fullfile(dataLAZPath,'*.laz'));

allFileNames = cell2mat({dataFolderLAZ.name});

noCatch = true;
cFolder = path1Server;
ftpobj = [];

for ii=1:size(filePathsAndNames{1},1)
    
    if( ~isempty( filePathsAndNames{1}{ii} ) )
        
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
                
                if(noCatch)
                    noCatch = false;
                    disp("At least one file is missing. Log in to download the files.")
                    
                    UserName = input('Write user name: \n','s');
                    Password = input('Write password: \n','s');
                    
                    ftpobj = ftp(pathWeb,UserName,Password);
                    
                    cFolder = [path1Server,filePathsAndNames{1}{ii}];
                    cd(ftpobj,cFolder);
                end

                disp(['Download: ',filePathsAndNames{2}{ii}(jj,:)])
                mget(ftpobj,filePathsAndNames{2}{ii}(jj,:),dataLAZPath);
                
            end
        end
    end
end

if (noCatch == false)
    close(ftpobj);
end


% UserName = input('Write user name','s');
% Password = input('Write password','s');
% 
% ftpobj = ftp(pathWeb,UserName,Password);
% cd(ftpobj,pathData);
% mget(ftpobj,dataName); % '19B030_65825_4150_25.laz'
% close(ftpobj);