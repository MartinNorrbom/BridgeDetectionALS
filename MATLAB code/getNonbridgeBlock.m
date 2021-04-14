function [blockCoord,intensityBlock,returnNumberBlock,pointLabel,blockLabel,blockGeoCoord] = ...
    getNonbridgeBlock(ptCloud,pointAttributes,class,tileBlockPointNumber,gridSize)
%  This function is used to get the tile block without bridge points from the
%  laz file.
%   Detailed explanation goes here:
%   Input: 
%       ptCloud: point cloud data
%       pointAttributes: the attributes of point cloud
%       class: the class of points
%       tileBlockPointNumber: number of points in unit tile block
%       gridSize: the size of time block (m x m)
%   Output:
%       blockCoord: zero Centered XYZ coordinates (the location) of points 
%                   in tile block
%       intensityBlock: the intensity of the points in tile block
%       returnNumberBlock: return numbers of the points in tile block

% Find x,y limits for all grids(tile block) of the input point cloud data
x = ptCloud.XLimits(1):gridSize:ptCloud.XLimits(2);
y = ptCloud.YLimits(1):gridSize:ptCloud.YLimits(2);
% Set x,y limits as a matrix
[X,Y] = meshgrid(x,y); 

numberOfBlock = (length(x)-1)*(length(y)-1);

% Create a 3 x pointsInOneTileBlock x totalTileBlocks empty matrix
blockCoord = single(zeros([3 tileBlockPointNumber numberOfBlock]));
returnNumberBlock = single(zeros([1 tileBlockPointNumber numberOfBlock]));
intensityBlock = single(zeros([1 tileBlockPointNumber numberOfBlock]));

blockGeoCoord = zeros([numberOfBlock 2]);

% Create a empty array for tile block class 
tileBlockClass = false([numberOfBlock,1]);

currentPoints = ptCloud.Location; 

    for ii=1:(length(x)-1)
        
        for jj = 1: (length(y)-1)
            % Find points in the block limit
             pointsInBlockLimit = ...
            (currentPoints(:,1)<=X(ii,jj+1)) & (currentPoints(:,1)> X(ii,jj))& ...
            (currentPoints(:,2)<=Y(ii+1,jj)) & (currentPoints(:,2)>Y(ii,jj));
             centerCoord = [(X(ii,jj) + gridSize/2), (Y(ii,jj) + gridSize/2)];
            
            % Get the classes of point in the block limit
            numberOfClass = pointAttributes.Classification(find(pointsInBlockLimit),:);

            if ~any(numberOfClass(:)==class)

                numberOfPointsInBlock = sum(pointsInBlockLimit);

                if( numberOfPointsInBlock<=0 )
                    % If there are no points in this tile block. Which can
                    % occure when the entire tile block is in a water area.
                    tileBlockClass( ((length(x)-1)*(ii-1)+jj) ) = true;
                    
                elseif(numberOfPointsInBlock< tileBlockPointNumber)      
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
                
                    blockCoord(:,:,((length(x)-1)*(ii-1)+jj)) = ptCloud.Location(randomSample,:)';
                    
                    % Zero center the blockCoor
                    blockCoord(:,:,((length(x)-1)*(ii-1)+jj)) = zeroCenteringTileBlock( blockCoord(:,:,((length(x)-1)*(ii-1)+jj)),centerCoord);
                    returnNumberBlock(:,:,((length(x)-1)*(ii-1)+jj)) = pointAttributes.LaserReturns(randomSample,:)';
                    intensityBlock(:,:,((length(x)-1)*(ii-1)+jj)) = ptCloud.Intensity(randomSample,:)';
                    
                    blockGeoCoord(((length(x)-1)*(ii-1)+jj),:) = [X(ii,jj+1)-gridSize/2,Y(ii+1,jj)-gridSize/2];
                    
            else
                tileBlockClass( ((length(x)-1)*(ii-1)+jj) ) = true;
            end
        
        end
    end 
    
    % Remove the bridge block 
    blockCoord(:,:,tileBlockClass) = [];
    returnNumberBlock(:,:,tileBlockClass) = [];
    intensityBlock(:,:,tileBlockClass) = [];
    
    % Return the greografic location of the tile block.
    blockGeoCoord(tileBlockClass,:) = [];

    
    % Set point- and block label
    pointLabel = single(false([tileBlockPointNumber numberOfBlock-length(find(tileBlockClass))]));
    blockLabel = single(zeros([1 numberOfBlock-length(find(tileBlockClass))]));
    
    

    
end

