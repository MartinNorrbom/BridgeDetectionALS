function [fileNames,fileForCoordinates,varargout] = getLAZFileFromCoord(cordinateList, dataInfo,varargin)
%fileForCoordinates return which index boxes and LAZ-files that are needed
% to generate tile blocks coordinates in the input "cordinateList". It also
% return information of where the files is located in "lantmäteriet's" server.

%   Input: 
%       cordinateList: Contains the coordinates for the index boxes that 
%           should be found.
%       dataInfo: Provides info of the quality class, paths and file names 
%           for the index boxes available.
%   Extra Inputs:
%       'neighbours': Include neighbouring index boxes for the boxes within
%           the corddinates of input "cordinateList".
%   Output:
%       fileNames: Returns the folder and the file names to "lantmäteriet's"
%           server where the LAZ files is located.
%       fileForCoordinates: Returns the last part of the LAZ-file names and
%           group the coordinates in input "cordinateList" to the LAZ-file
%           there are located in.
%   Extra Output:
%       varargout{1}: Returns the names of the neighbouring LAZ files to
%           with the same indexing as "fileForCoordinates".


    % Get the coordinate in text format.
    coordinatesTextFormat = [{num2str(cordinateList(:,1))},{num2str(cordinateList(:,2))}];

    % Get the names of the index blocks.
    [~,LAZfileNames2] = getNameOfIndexBlock(coordinatesTextFormat);

    % Get the unique index blocks.
    [uLAZfileNames2,uniqueIndex,allIndex] = unique(LAZfileNames2,'rows','stable');
    

    
    if(nargin >= 3)
        % If detection neighbouring index boxes is enable.
        if( contains(varargin{1},"neighbours") )
            
            % Get neighbours of index blocks.
            LAZfileNeighbourNames = getNeighbourFiles(LAZfileNames2);
            
            % Get the unique index box including neighbours.
            fileNamesIncludingNeibours = unique( [uLAZfileNames2;cell2mat(LAZfileNeighbourNames)],'rows','stable' );
            uNorthing = str2double(string(fileNamesIncludingNeibours(:,2:6)))*100;
            uEasting = str2double(string(fileNamesIncludingNeibours(:,8:11)))*100;
            
            % A list of the needed index boxes.
            LAZfilesToGet = fileNamesIncludingNeibours;
            
            if(nargout>2)
                varargout{1} = LAZfileNeighbourNames;
            end
        else
            error("Wrong input argument (3).");
        end
    else

        % Get unique coordinates for unqiue index boxs.
        uNorthing = cordinateList(uniqueIndex,1);
        uEasting = cordinateList(uniqueIndex,2);
        % A list of the needed index boxes, without neighbours.
        LAZfilesToGet = LAZfileNames2;
        
    end


    
    
    % Map each of the coordinates to one index box.
    fileForCoordinates = cell(length(uniqueIndex),2);
    for ii=1:length(uniqueIndex)
        fileForCoordinates{ii,1} = uLAZfileNames2(ii,2:end);
        fileForCoordinates{ii,2} = cordinateList(allIndex==ii,:);
    end
    
    
    indexesWithClass3 = find([ dataInfo.KLASS ] == 3);
    % Get the number of regions.
    numberOfRegionsClass3 = length(indexesWithClass3);
    
    % Store points that are in the region.
    regionPath = cell(numberOfRegionsClass3,1);
    % Store file name of the 
    regionLAZfileNames = cell(numberOfRegionsClass3,1);
    
    % Get region and time of sampling for index blocks.
    for ii=1:numberOfRegionsClass3
    
        % Get index of area with class 3.
        jj = indexesWithClass3(ii);
        % Find which region the cordinates is located in.
        blocksInRegion = ...
            ( (min(dataInfo(jj).BoundingBox(:,2)) <= uNorthing) &  ...
            (uNorthing < max(dataInfo(jj).BoundingBox(:,2))) ) & ...
            ( (min(dataInfo(jj).BoundingBox(:,1)) <= uEasting) &  ...
            (uEasting < max(dataInfo(jj).BoundingBox(:,1))) );
        
        % Check if there was an available region the coordinates was
        % located in.
        if sum(blocksInRegion) > 0
            
            % Create a path to the region.
            regionInfo = dataInfo(jj).OMR0xC5DE;
            regionPath{ii} = ['/20',regionInfo(1:2),'/',regionInfo,'/'];
            
            % Indicates which region the index boxes is located in.
            regionLAZfileNames{ii} = ...
                [repmat(regionInfo,[sum(blocksInRegion),1]),LAZfilesToGet(blocksInRegion,:)];
        end
    end
    
    % Return path and file names.
    fileNames = {regionPath,regionLAZfileNames};

end


function [nameOfIndexBlocks,LAZfileNames2] = getNameOfIndexBlock(cordinateList)
% This function returns the names of index block from a list of coordinates.
% This function follows the rules that is descibed in this webpage:
% https://www.lantmateriet.se/globalassets/kartor-och-geografisk-information/gps-och-geodetisk-matning/info_blad-11.pdf

    numberOfCoord = size(cordinateList{1},1);

    Northing = cordinateList{1};
    Easting = cordinateList{2};
    
    % Get an index block of 10x10 Km.
    Northing10Block = Northing(:,1:3);
    Easting10Block = Easting(:,1:2);
    
    % Get 2.5x2.5 Km block inside of the 10x10 Km.
    Northing2_5Block = Northing(:,4:5);
    Easting2_5Block = Easting(:,3:4);
    
    % Get the for last letters of the index boxes name.
    Northing10Block2_5 = repmat('00',[numberOfCoord,1]);
    Easting10Block2_5 = repmat('00',[numberOfCoord,1]);
        
    NInd25 = ((25 <= str2num(Northing2_5Block) ) & ( str2num(Northing2_5Block) < 50) );
    NInd50 = ( (50 <= str2num(Northing2_5Block) ) & ( str2num(Northing2_5Block) < 75) );
    NInd75 = (75 <= str2num(Northing2_5Block));
    
    EInd25 = ((25 <= str2num(Easting2_5Block) ) & ( str2num(Easting2_5Block) < 50) );
    EInd50 = ( (50 <= str2num(Easting2_5Block) ) & ( str2num(Easting2_5Block) < 75) );
    EInd75 = (75 <= str2num(Easting2_5Block));
    
    if(sum(NInd25)>0)
        Northing10Block2_5(NInd25,1:2) = repmat('25',[sum(NInd25),1]);
    end
    
    if(sum(NInd50)>0)
        Northing10Block2_5(NInd50,1:2) = repmat('50',[sum(NInd50),1]);
    end
    
    if(sum(NInd75)>0)
        Northing10Block2_5(NInd75,1:2) = repmat('75',[sum(NInd75),1]);
    end
    
    if(sum(EInd25)>0)
        Easting10Block2_5(EInd25,1:2) = repmat('25',[sum(EInd25),1]);
    end
    
    if(sum(EInd50)>0)
        Easting10Block2_5(EInd50,1:2) = repmat('50',[sum(EInd50),1]);
    end
    
    if(sum(EInd75)>0)
        Easting10Block2_5(EInd75,1:2) = repmat('75',[sum(EInd75),1]);
    end
    
    % Generate as much underscores as there are coordinates.
    underScoreVector = repmat('_',[numberOfCoord,1]);
    
    % Get the names of the index boxes.
    nameOfIndexBlocks = [Northing10Block,underScoreVector,Easting10Block, ...
        underScoreVector,Northing10Block2_5,Easting10Block2_5];
    
    % Get last part of the file name for LAZ-file data.
    LAZfileNames2 = [underScoreVector,Northing10Block,Northing10Block2_5, underScoreVector, ...
        Easting10Block,Easting10Block2_5,repmat('_25.laz',[numberOfCoord,1])];

end


function [LAZfileNeighbourNames] = getNeighbourFiles(LAZfileNames)
%getNeighbourFiles returns the names neighbours to the list index blocks in 
% "LAZfileNames".
%
%   Input: 
%       LAZfileNames: Is the list of index boxes.
%   Output:
%       LAZfileNeighbourNames is an cell array where each cell contains the 
%       name of neighbouring index blocks for the index block with the same 
%       index in "LAZfileNames".

    % The columns that contain the northing numbers.
    northStrIndex = 2:6;
    % The columns that contain the easting numbers.
    eastStrIndex = 8:11;

    % Convert the coordinates in text format to numbers.
    Northing = str2double(LAZfileNames(:,northStrIndex));
    Easting = str2double(LAZfileNames(:,eastStrIndex));
    
    % How the neighbouring file differs in name. See link in function
    % "getNameOfIndexBlock" in this matlab file.
    neigbourNameDiffN = [-25,0,25,-25,25,-25,0,25]';
    neigbourNameDiffE = [-25,-25,-25,0,0,25,25,25]';
    
    % Allocate cells to store the neighbours names.
    LAZfileNeighbourNames = cell(size(LAZfileNames,1),1);
    
    % Loop through all index boxes in "LAZfileNames" and find the
    % neighbours.
    for ii=1:size(LAZfileNames,1)
        % Get the coordinates/names of the neighbouring index boxes.
        NorthingNeighbours = Northing(ii) + neigbourNameDiffN;
        EastingNeighbours = Easting(ii) + neigbourNameDiffE;
        
        % Allocate the same number of file names as neighbours(LAZ-files).
        tempText = repmat(LAZfileNames,length(neigbourNameDiffN),1);
        
        % Change the file name to the neighbouring index boxes(LAZ-files).
        tempText(:,northStrIndex) = num2str(NorthingNeighbours);
        tempText(:,eastStrIndex) = num2str(EastingNeighbours);
        
        % Store the names of the neighbours.
        LAZfileNeighbourNames{ii} = tempText;
        
    end
    
end