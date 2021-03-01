function [fileNames,fileForCoordinates] = getLAZFileFromCoord(cordinateList, dataInfo)
%fileForCoordinates return which index boxes and LAZ-files that are needed
%for the "cordinateList".

%   cordinateList - Is to find which index boxes that are needed.
%   dataInfo      - Provides info of paths and file names for the index
%                   boxes available.

    % Get the coordinate in text format.
    coordinatesTextFormat = [{num2str(cordinateList(:,1))},{num2str(cordinateList(:,2))}];

    % Get the names of the index blocks.
    [~,LAZfileNames2] = getNameOfIndexBlock(coordinatesTextFormat);

    % Get the unique index blocks.
    [uLAZfileNames2,uniqueIndex,allIndex] = unique(LAZfileNames2,'rows','stable');
    
    % Get unique coordinates for unqiue index boxs.
    uNorthing = cordinateList(uniqueIndex,1);
    uEasting = cordinateList(uniqueIndex,2);
    
    % Map each of the coordinates to one index box.
    fileForCoordinates = cell(length(uniqueIndex),2);
    for ii=1:length(uniqueIndex)
        fileForCoordinates{ii,1} = uLAZfileNames2(ii,2:end);
        fileForCoordinates{ii,2} = cordinateList(allIndex==ii,:);
    end
    
    % Get the number of regions.
    numberOfRegions = length(dataInfo);
    
    % Store points that are in the region.
    regionPath = cell(numberOfRegions,1);
    % Store file name of the 
    regionLAZfileNames = cell(numberOfRegions,1);
    
    % Get region and time of sampling for index blocks.
    for ii=1:numberOfRegions
    
        % Find which region the cordinates is located in.
        blocksInRegion = ...
            ( (min(dataInfo(ii).BoundingBox(:,2)) <= uNorthing) &  ...
            (uNorthing < max(dataInfo(ii).BoundingBox(:,2))) ) & ...
            ( (min(dataInfo(ii).BoundingBox(:,1)) <= uEasting) &  ...
            (uEasting < max(dataInfo(ii).BoundingBox(:,1))) );
        
        % Check if there was an available region the coordinates was
        % located in.
        if sum(blocksInRegion) > 0
            
            % Create a path to the region.
            regionInfo = dataInfo(ii).OMR0xC5DE;
            regionPath{ii} = ['/20',regionInfo(1:2),'/',regionInfo,'/'];
            
            % Indicates which region the index boxes is located in.
            regionLAZfileNames{ii} = ...
                [repmat(regionInfo,[sum(blocksInRegion),1]),uLAZfileNames2(blocksInRegion,:)];
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

