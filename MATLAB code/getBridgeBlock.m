function [blockCoord,intensityBlock,returnNumberBlock,pointLabel,blockLabel,blockGeoCoord] = ...
    getBridgeBlock(ptCloud,pointAttributes,class,tileBlockPointNumber,gridSize)
%  This function is used to get the tile block with bridge points from the
%  laz file.   
%   Detailed explanation goes here:
%   Input: 
%       ptCloud: point cloud data
%       pointAttributes: the attributes of point cloud
%       class: the class of points
%       tileBlockPointNumber: number of points in unit tile block
%       gridSize: the size of time block (m x m)
%   Output:
%       dataSetSkog: XYZ coordinates (the location) of points in tile block
%       intensityBlock: the intensity of the points in tile block
%       returnNumberBlock: return numbers of the points in tile block


bridgePoints = pointAttributes.Classification == class;

xyzBridgePoints = ptCloud.Location(bridgePoints,:);

numberOfBlock = ceil(2*((ptCloud.XLimits(2)-ptCloud.XLimits(1))/gridSize)*((ptCloud.YLimits(2)-ptCloud.YLimits(1))/gridSize));

% Create a 3 x pointsInOneTileBlock x totalTileBlocks empty matrix
blockCoord = single(zeros([3 tileBlockPointNumber numberOfBlock]));
returnNumberBlock = single(zeros([1 tileBlockPointNumber numberOfBlock]));
intensityBlock = single(zeros([1 tileBlockPointNumber numberOfBlock]));
pointLabel = single(zeros([tileBlockPointNumber numberOfBlock]));

blockGeoCoord = zeros([numberOfBlock 2]);

currentBridgePoints = xyzBridgePoints; 

% figure(5)

ii=0;

while ~isempty(currentBridgePoints)
    ii = ii+1;
    
    % To prevent from always having a bridge point at center of the center 
    % of a tuke block, there will be a random offset from where the 
    % generated tile block will be located. The current bridge point will
    % always be within a square that 90% of the tile block closest to the 
    % center of the generated tile block. 
    randomOffset = (rand(1,2)-0.5)*(gridSize*0.75); 
    
    % The center of the current tile block.
    blockGeoCoord(ii,:) = currentBridgePoints(1,1:2)+randomOffset;
    
    % Set limits of the current tile block.
    xLim = blockGeoCoord(ii,1)+[0.5,-0.5]*gridSize; %+-50m in x axis from the current point
    yLim = blockGeoCoord(ii,2)+[0.5,-0.5]*gridSize; %+-50m in y axis from the current point
    
    % Select the bridge points that lie inside the tile block
    bridgeInBlockLimit = ...
    (currentBridgePoints(:,1)<=xLim(1)) & (currentBridgePoints(:,1)>xLim(2))& ...
    (currentBridgePoints(:,2)<=yLim(1)) & (currentBridgePoints(:,2)>yLim(2));

    % remove selected bridge points from currentBridgepoints.
    currentBridgePoints(bridgeInBlockLimit,:) = []; 
    
    % Select all points that lie inside the tile block
    pointsInBlockLimit = ...
    (ptCloud.Location(:,1)<=xLim(1)) & (ptCloud.Location(:,1)>xLim(2))& ...
    (ptCloud.Location(:,2)<=yLim(1)) & (ptCloud.Location(:,2)>yLim(2));

    %pcshow(ptCloud.Location(pointsInBlockLimit,:));
    %w = waitforbuttonpress;
    
    
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
    
    % Store tile blocks from randomly sampled points
    blockCoord(:,:,ii) = ptCloud.Location(randomSample,:)';
    
    blockCoord(:,:,ii) = zeroCenteringTileBlock( blockCoord(:,:,ii),blockGeoCoord(ii,:));
    returnNumberBlock(:,:,ii) = pointAttributes.LaserReturns(randomSample,:)';
    intensityBlock(:,:,ii) = ptCloud.Intensity(randomSample,:)';
   
    % Get point label for bridge point
    pointLabel(:,ii) = pointAttributes.Classification(randomSample,:)'==17;
    
    
end
    % Set all tile block labels to positiv.
    blockLabel = single(ones([1 ii]));

    % Remove the unallocated data.
    blockCoord(:,:,(ii+1):end) = [];
    returnNumberBlock(:,:,(ii+1):end) = [];
    intensityBlock(:,:,(ii+1):end) = [];
    pointLabel(:,(ii+1):end) = [];
    
    blockGeoCoord((ii+1):end,:) = [];
    
end

