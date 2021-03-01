function [dataSetSkog,returnNumberBlock,intensityBlock,pointLabel,blockLabel] = ...
    getBlockFromCoord(ptCloud,pointAttributes,class,tileBlockPointNumber,gridSize, coordinates,varargin)
%getBlockFromCoord generates tile blocks from selected coordinates. It set
% the current coordinate in the list to the middle of the tile block, then
% if a coordinate was within a tile block that was generated earlier it
% skip to generate another tile block for this coordinate.

%   Input: 
%       ptCloud: point cloud data
%       pointAttributes: the attributes of point cloud
%       class: the class of points
%       tileBlockPointNumber: number of points in unit tile block
%       gridSize: the size of time block (m x m)
%       coordinates: the list of coordinates where tile blocks are going to
%                   be generated.
%   Output:
%       dataSetSkog: XYZ coordinates (the location) of points in tile block
%       intensityBlock: the intensity of the points in tile block
%       returnNumberBlock: return numbers of the points in tile block
%       pointLabel: Label for each point in the tile blocks. 0 is non
%                   bridge 1 is bridge.
%       blockLabel: Label for each tile blocks, 0 is non bridge 1 is bridge.

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
            % Count the number of generated tile blocks.
            ii = ii+1;

            % Get the limits of the tile block, with the selected
            % coordinate in the middle.
            xLim = coordinates(coordCount,1)+[1,-1]*gridSize; 
            yLim = coordinates(coordCount,2)+[1,-1]*gridSize; 

            % Select the coordinates that lie inside the tile block.
            coordInBlockLimit = ...
            (coordinates(:,1)<=xLim(1)) & (coordinates(:,1)>xLim(2))& ...
            (coordinates(:,2)<=yLim(1)) & (coordinates(:,2)>yLim(2));

            % remove coordinates that are within the tile block.
            coordinateCheckList(coordInBlockLimit) = true; 

            % Select all points that lie inside the tile block
            pointsInBlockLimit = ...
            (ptCloud.Location(:,1)<=xLim(1)) & (ptCloud.Location(:,1)>xLim(2))& ...
            (ptCloud.Location(:,2)<=yLim(1)) & (ptCloud.Location(:,2)>yLim(2));

            % Get the number of points in the tile block.
            numberOfPointsInBlock = sum(pointsInBlockLimit);

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
            dataSetSkog(:,:,ii) = ptCloud.Location(randomSample,:)';
            returnNumberBlock(:,:,ii) = pointAttributes.LaserReturns(randomSample,:)';
            intensityBlock(:,:,ii) = ptCloud.Intensity(randomSample,:)';

            % Get point label for bridge point.
            pointLabel(:,ii) = pointAttributes.Classification(randomSample,:)'==class;
            
            % Label the whole block.
            if( sum(pointLabel(:,ii)) <= 0  )
                blockLabel(1,ii) = 0;
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

    if(nargin > 6)
        
        if( contains( varargin{1},"RemovePositive" ) )
            % Removes bridges
            indexToRemove = blockLabel==1;
            
            blockLabel(:,indexToRemove) = [];
            dataSetSkog(:,:,indexToRemove) = [];
            returnNumberBlock(:,:,indexToRemove) = [];
            intensityBlock(:,:,indexToRemove) = [];
            
            pointLabel(:,indexToRemove) = [];
        end
    
    end
    
end

