function [dataSetSkog,intensityBlock,returnNumberBlock,pointLabel,blockLabel] = ...
    getBlockFromCoord(ptCloud,pointAttributes,class,tileBlockPointNumber,gridSize, coordinates,varargin)
%getBlockFromCoord generates tile blocks from selected coordinates. It set
% the current coordinate in the list to the middle of the tile block, then
% if a coordinate was within a tile block that was generated earlier it
% skip to generate another tile block for that coordinate. It can also be
% set to enable buffering of neighbouring index boxes and remove the tile
% blocks that contains bridges.

%   Input: 
%       ptCloud: point cloud data
%       pointAttributes: the attributes of point cloud
%       class: the class of points
%       tileBlockPointNumber: number of points in unit tile block
%       gridSize: the size of time block (m x m)
%       coordinates: the list of coordinates where tile blocks are going to
%                   be generated.
%   Extra Inputs:
%       'neighbours': will enable buffering of neighbouring index, so that
%           tile blocks that are close to the edges of the main blocks will use
%           points from the neighbouring blocks.
%       'RemovePositive': will remove tile blocks that contains bridges.
%   Output:
%       dataSetSkog: XYZ coordinates (the location) of points in tile block
%       intensityBlock: the intensity of the points in tile block
%       returnNumberBlock: return numbers of the points in tile block
%       pointLabel: Label for each point in the tile blocks. 0 is non
%                   bridge 1 is bridge.
%       blockLabel: Label for each tile blocks, 0 is non bridge 1 is bridge.


    % Count the number of extra inputs.
    extraInput = nargin - 6;
    % Allocate array to indicate if an feature should be on or off.
    extraFeature = false(2,1);
    
    % If there is at least one extra input
    if(extraInput > 0)
        % Loop through all extra input.
        for ii=1:extraInput
            % Check what type of feature that should be enabled.
            if( contains( varargin{ii},"neighbours" ) )
                extraFeature(1) = true;
            elseif( contains( varargin{ii},"RemovePositive" ) )
                extraFeature(2) = true;
            else
                error(['Wrong input argument (',num2str(ii+6),').']);
            end
        end
    end
    
    % If buffering of neighbouring index boxes is enable.
    if(extraFeature(1))
        % Get the limits of the first block (the main block).
        xLim = ptCloud{1}.XLimits;
        yLim = ptCloud{1}.YLimits;
        
        % Get the number of neighbouring index blocks.
        numberOfSets = length(ptCloud);
        
        % Allocate temporary buffers for the extra points.
        tempBuffCoords = cell(numberOfSets,1);
        tempBuffIntensity = cell(numberOfSets,1);
        tempBuffClass = cell(numberOfSets,1);
        tempBuffLaserReturn = cell(numberOfSets,1);

        % Loop through all index blocks and save the points within the zoon
        % where tile blocks can be generated.
        for ii=1:numberOfSets
            
            % Get index of points that are within the zoon. The buffert
            % size of neighbouring blocks is those that are within the
            % distance of the gridSize to the main block in [x,y]
            % coordinates.
            ptIndex = ...
            ( min(xLim)-gridSize < ptCloud{ii}.Location(:,1) ) & ...
            ( ptCloud{ii}.Location(:,1) < max(xLim)+gridSize ) & ...
            ( min(yLim)-gridSize < ptCloud{ii}.Location(:,2) ) & ...
            ( ptCloud{ii}.Location(:,2) < max(yLim)+gridSize );
        
            % Buffer all the point features for the points within the zoon
            tempBuffCoords{ii} = ptCloud{ii}.Location(ptIndex,:);
            tempBuffIntensity{ii} = ptCloud{ii}.Intensity(ptIndex);
            tempBuffClass{ii} = pointAttributes{ii}.Classification(ptIndex);
            tempBuffLaserReturn{ii} = pointAttributes{ii}.LaserReturns(ptIndex);
            
        end
        % Merge all the point features to arrays.
        pointCoords = cell2mat(tempBuffCoords);
        pointIntensity = cell2mat(tempBuffIntensity);
        pointClass = cell2mat(tempBuffClass);
        pointLaserReturn = cell2mat(tempBuffLaserReturn);
        
    else
        % If no neighbouring index boxes should be included, only use the
        % point features in the main block.
        pointCoords = ptCloud.Location;
        pointIntensity = ptCloud.Intensity;
        pointClass = pointAttributes.Classification;
        pointLaserReturn = pointAttributes.LaserReturns;
    end

    % Maximum number of tile blocks.
    numberOfBlock = size(coordinates,1);

    % Create a list to indicate if each coordinate has a tile block.
    coordinateCheckList = false(numberOfBlock,1);
    
    % Allocate space to return tile block data.
    dataSetSkog = single(zeros([3 tileBlockPointNumber numberOfBlock]));
    returnNumberBlock = single(zeros([1 tileBlockPointNumber numberOfBlock]));
    intensityBlock = single(zeros([1 tileBlockPointNumber numberOfBlock]));
    pointLabel = single(zeros([tileBlockPointNumber numberOfBlock]));
    blockLabel = single(ones([1 numberOfBlock]));

    % Loops until all tile blocks are generated.
    ii=0;
    coordCount = 0;
    while sum(coordinateCheckList) < numberOfBlock
        
        % Count cordinate list.
        coordCount = coordCount+1;
        if(coordinateCheckList(coordCount) == false)

            % Get the limits of the tile block, with the selected
            % coordinate in the middle.
            xLim = coordinates(coordCount,1)+[0.5,-0.5]*gridSize; 
            yLim = coordinates(coordCount,2)+[0.5,-0.5]*gridSize; 

            % Select the coordinates that lie inside the tile block.
            coordInBlockLimit = ...
            (coordinates(:,1)<=xLim(1)) & (coordinates(:,1)>xLim(2))& ...
            (coordinates(:,2)<=yLim(1)) & (coordinates(:,2)>yLim(2));

            % remove coordinates that are within the tile block.
            coordinateCheckList(coordInBlockLimit) = true; 

            % Select all points that lie inside the tile block
            pointsInBlockLimit = ...
            (pointCoords(:,1)<=xLim(1)) & (pointCoords(:,1)>xLim(2))& ...
            (pointCoords(:,2)<=yLim(1)) & (pointCoords(:,2)>yLim(2));

            % Get the number of points in the tile block.
            numberOfPointsInBlock = sum(pointsInBlockLimit);

            if( numberOfPointsInBlock > 0 )
            % If there are no points in the tile block this step will be scipped. 
            % Tile blocks with no points can occure when the entire tile 
            % block is in a water area.
            
                % Count the number of generated tile blocks.
                ii = ii+1;
            
                if(numberOfPointsInBlock< tileBlockPointNumber)      
                    % If the number of points in the tile block are less than tileBlockPointNumber,
                    % interpolate the number to tileBlockPointNumber by copying existing points.

                    % Get the number of missing points
                    numberOfDublicates = tileBlockPointNumber - numberOfPointsInBlock;  

                    % Increase the number of points to the specified number for each
                    % tile block.
                    dublicates = datasample(find(pointsInBlockLimit),numberOfDublicates);

                    % Merge all points with dublicated points. 
                    randomSample = [find(pointsInBlockLimit);dublicates];

                    % Random permutation of the points in whole tile block
                    randomSample = randomSample(randperm(length(randomSample)));

                else
                    % Randomly sample tileBlockPointNumber points from the tiles block 
                    randomSample = datasample(find(pointsInBlockLimit),tileBlockPointNumber);
                end

                % Save the point features for the points in the tile block.
                dataSetSkog(:,:,ii) = pointCoords(randomSample,:)';
                returnNumberBlock(:,:,ii) = pointLaserReturn(randomSample,:)';
                intensityBlock(:,:,ii) = pointIntensity(randomSample,:)';

                % Get point label for bridge point.
                pointLabel(:,ii) = pointClass(randomSample,:)'==class;

                % Label the whole block.
                if( sum(pointLabel(:,ii)) <= 0  )
                    blockLabel(1,ii) = 0;
                end
                
            end

        end
        
    end
    
    % Remove empty index.
    if(ii < numberOfBlock)
        blockLabel(:,(ii+1):end) = [];
        dataSetSkog(:,:,(ii+1):end) = [];
        returnNumberBlock(:,:,(ii+1):end) = [];
        intensityBlock(:,:,(ii+1):end) = [];
        pointLabel(:,(ii+1):end) = [];
    end

    % If removal of bridges is enabled.
    if( extraFeature(2) )
        % Get indecies of tile blocks that contains bridges.
        indexToRemove = blockLabel==1;

        % Remove all data in tile blocks that contains bridges.
        blockLabel(:,indexToRemove) = [];
        dataSetSkog(:,:,indexToRemove) = [];
        returnNumberBlock(:,:,indexToRemove) = [];
        intensityBlock(:,:,indexToRemove) = [];
        pointLabel(:,indexToRemove) = [];
    end
    
end

