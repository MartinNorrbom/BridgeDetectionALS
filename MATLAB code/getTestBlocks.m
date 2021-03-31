function [blockCoord,intensityBlock,returnNumberBlock,pointLabel,blockLabel,blockGeoCoord] = ...
    getTestBlocks(ptCloud,pointAttributes,class,tileBlockPointNumber,gridSize,overlap)
%getTestBlocks Return tile blocks in a grid shape. Overlapping can be
%specified.

%   Input: 
%       ptCloud: point cloud data
%       pointAttributes: the attributes of point cloud
%       class: the class of points
%       tileBlockPointNumber: number of points in unit tile block
%       gridSize: the size of time block (m x m)
%       coordinates: the list of coordinates where tile blocks are going to
%                   be generated.
%       overlap: Specify the "precentage length" of the "gridSize" between 
%           tile block. Should be a value from zero and lower than 1.
%   Output:
%       blockCoord: XYZ coordinates (the location) of points in tile block
%       intensityBlock: the intensity of the points in tile block
%       returnNumberBlock: return numbers of the points in tile block
%       pointLabel: Label for each point in the tile blocks. 0 is non
%                   bridge 1 is bridge.
%       blockLabel: Label for each tile blocks, 0 is non bridge 1 is bridge.

    
    % If buffering of neighbouring index boxes (LAZ-files) is enable.
    if(length(ptCloud) > 1)
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
        
        % Get the limits of the point cloud.
        xLim = ptCloud.XLimits;
        yLim = ptCloud.YLimits;
        
        % If no neighbouring index boxes should be included, only use the
        % point features in the main block.
        pointCoords = ptCloud.Location;
        pointIntensity = ptCloud.Intensity;
        pointClass = pointAttributes.Classification;
        pointLaserReturn = pointAttributes.LaserReturns;
    end
    
    % Find the distance between each tile block.
    lengthBetweenTileBlocks = gridSize*(1-overlap);
    
    % Get start x and y coordinates for the tile blocks.
    xStart = min(xLim)+lengthBetweenTileBlocks/2;
    yStart = min(yLim)+lengthBetweenTileBlocks/2;
    % Get end x and y cooordinates for the tile blocks.
    xEnd = max(xLim);
    yEnd = max(yLim);

    
    % Get center coordinates for each tile block.
    x = xStart:lengthBetweenTileBlocks:xEnd;
    y = yStart:lengthBetweenTileBlocks:yEnd;
    
    % Get the number of tile blocks.
    nrTileBlocks = length(x)*length(y);
    
    % Allocate variables to store the info of the generated tile blocks.
    blockCoord = single(zeros([3 tileBlockPointNumber nrTileBlocks]));
    returnNumberBlock = single(zeros([1 tileBlockPointNumber nrTileBlocks]));
    intensityBlock = single(zeros([1 tileBlockPointNumber nrTileBlocks]));
    pointLabel = single(false([tileBlockPointNumber nrTileBlocks]));
    blockLabel = single(zeros([1 nrTileBlocks]));
    blockGeoCoord = zeros([nrTileBlocks 2]);
    
    % To check if there is any tile block that do not contain any points.
    invalidTileBlocks = false(nrTileBlocks,1);
    
    
    % Generate the tile blocks.
    for ii=1:nrTileBlocks
        
        % Get the x and y coordinate for the current tile block.
        curXind = mod((ii-1),length(x))+1;
        curYind = ceil(ii/length(x));
        
        % Get the limits in x and y for the current tile block.
        xlimTileBlock = x(curXind)+[0.5,-0.5]*gridSize;
        ylimTileBlock = y(curYind)+[0.5,-0.5]*gridSize;
        
        % Get all points within the limit.
        pointsInBlockLimit = ...
            (pointCoords(:,1)<=xlimTileBlock(1)) & (pointCoords(:,1)> xlimTileBlock(2))& ...
            (pointCoords(:,2)<=ylimTileBlock(1)) & (pointCoords(:,2)> ylimTileBlock(2));
        
        numberOfPointsInBlock = sum(pointsInBlockLimit);
        
        % Check if there are any blocks within the limit.
        if( numberOfPointsInBlock<=0 )
            
            invalidTileBlocks(ii) = true;
            
        else
            % Check if the are more or less points in the tile block than
            % the desired number of points (tileBlockPointNumber).
            if(numberOfPointsInBlock< tileBlockPointNumber)      
                % If the number of points in the tile block are less than tileBlockPointNumber,
                % interpolate the number points to tileBlockPointNumber by copying existing points.

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
                % If there are equal of more existing points within the
                % tile block than the required "tileBlockPointNumber",
                % downsampling of points will be made.
                
                % Randomly sample tileBlockPointNumber points from the tiles block 
                randomSample = datasample(find(pointsInBlockLimit),tileBlockPointNumber);
            end

            % Zero center the [x,y,z] coordinates.
            blockCoord(:,:,ii) = zeroCenteringTileBlock( pointCoords(randomSample,:)', [x(curXind),y(curYind)]);
            % Store return number and intensity.
            returnNumberBlock(:,:,ii) = pointLaserReturn(randomSample,:)';
            intensityBlock(:,:,ii) = pointIntensity(randomSample,:)';

            % Save the center coordinates of the tile block.
            blockGeoCoord(ii,:) = [x(curXind),y(curYind)];
            
            % Label the current tile block and the points within the tile
            % block.
            numberOfClass = pointClass(randomSample);
            pointLabel(:,ii) = (numberOfClass == class);
            blockLabel(1,ii) = sum(numberOfClass == class) > 0;

        end
        
    end
    
    % Remove invalid tile blocks.
    blockCoord(:,:,invalidTileBlocks) = [];
    returnNumberBlock(:,:,invalidTileBlocks) = [];
    intensityBlock(:,:,invalidTileBlocks) = [];
    blockGeoCoord(invalidTileBlocks,:) = [];
    pointLabel(:,invalidTileBlocks) = [];
    blockLabel(:,invalidTileBlocks) = [];
    
end

