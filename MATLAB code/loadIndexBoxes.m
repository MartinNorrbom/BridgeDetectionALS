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
manualSelectedCoordinates = readCoordinates(CoordinatesPath,CoordinatesFileName,20);

%%
dataLocationInfo = shaperead([dataInfoPath,laserDataLocationInfo]);
%%
% Find the LAZ-files for the selected coordinates.
[filePathsAndNames,fileForCoordinates,neighbourFiles] = getLAZFileFromCoord(manualSelectedCoordinates, dataLocationInfo,"neighbours");

%%
% ------------- Download all the missing files ----------------

% % Get all the file names for the LAZ files located in the data folder.
% dataFolderLAZ = dir(fullfile(dataLAZPath,'*.laz'));
% allFileNames = cell2mat({dataFolderLAZ.name});
% 
% if(isempty(allFileNames))
%     allFileNames = '';
% end
% 
% noCatch = true;
% cFolder = path1Server;
% ftpobj = [];
% 
% % Loop through all the regions.
% for ii=1:size(filePathsAndNames{1},1)
%     
%     % Check if there are index boxes located in the current region.
%     if( ~isempty( filePathsAndNames{1}{ii} ) )
%         
%         % If at least one file is missing and the current index box is 
%         % missing, locate the path to the current index box in the server.
%         if (noCatch == false)
%             if ~contains(cFolder,[path1Server,filePathsAndNames{1}{ii}] )
%                 cFolder = [path1Server,filePathsAndNames{1}{ii}];
%                 cd(ftpobj,cFolder);
%             end
%         end
%         
%         lengthJJ = size(filePathsAndNames{2}{ii},1);
%         for jj=1:lengthJJ
% 
%             % Download the LAZ file if it is not downloaded.
%             if( ~contains(allFileNames,filePathsAndNames{2}{ii}(jj,:)) )
%                 
%                 % Check if an connection to the ftp server is establish.
%                 if(noCatch)
%                     noCatch = false;
%                     disp("At least one file is missing. Log in to download the files.")
%                     
%                     % Write user name and password for the ftp-server.
%                     UserName = input('Write user name: \n','s');
%                     Password = input('Write password: \n','s');
%                     
%                     % Connect and login to the ftp server
%                     ftpobj = ftp(pathWeb,UserName,Password);
%                     
%                     % Get to the folder in the ftp server where the missing 
%                     % index box is located.
%                     cFolder = [path1Server,filePathsAndNames{1}{ii}];
%                     cd(ftpobj,cFolder);
%                 end
%                 
%                 % Download missing index box
%                 disp(['Download: ',filePathsAndNames{2}{ii}(jj,:)])
%                 mget(ftpobj,filePathsAndNames{2}{ii}(jj,:),dataLAZPath);
%                 
%             end
%         end
%     end
% end
% 
% % Close ftp server connection.
% if (noCatch == false)
%     close(ftpobj);
% end

%%

% ------ Generate tile-blocks for the selected coordinates ------

% Parameters for tile-blocks
gridSize = 100;
tileBlockPointNumber = 20000;
class = 17;

generationMethod = zeros(size(fileForCoordinates,1),3);
generationMethod(:,[1,2,3]) = 1;

[coordBlock,intensityBlock,returnNumberBlock,pointLabel,blockLabel] = ... 
    generateTileBlocks(fileForCoordinates,generationMethod,gridSize, ...
    tileBlockPointNumber,class,"dataPath",dataLAZPath,"neighbours",neighbourFiles);

for ii=1:size(coordBlock,3)
    pcshow(coordBlock(:,:,ii)', intensityPlot(intensityBlock(1,:,ii),6))
    w = waitforbuttonpress;
end

% --- Make clusters to get the data more detailed ---
% GMModel = fitgmdist(dataSetSkog(:,:,jj)',5);
% idx = cluster(GMModel,dataSetSkog(:,:,jj)');

% --- Commands for the ftp-server ---
% ftpobj = ftp(pathWeb,UserName,Password);
% cd(ftpobj,pathData);
% mget(ftpobj,dataName);
% close(ftpobj);