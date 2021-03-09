function [dataSetSkogBridge,intensityBlock,returnNumberBlock,pointLabel,blockLabel] = ...
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

numberOfBlock = 2*((ptCloud.XLimits(2)-ptCloud.XLimits(1))/gridSize)*((ptCloud.YLimits(2)-ptCloud.YLimits(1))/gridSize);

% Create a 3x2048x1648 empty matrix
dataSetSkogBridge = single(zeros([3 tileBlockPointNumber numberOfBlock]));
returnNumberBlock = single(zeros([1 tileBlockPointNumber numberOfBlock]));
intensityBlock = single(zeros([1 tileBlockPointNumber numberOfBlock]));
pointLabel = single(zeros([tileBlockPointNumber numberOfBlock]));

currentBridgePoints = xyzBridgePoints; 

% figure(5)

ii=0;

while ~isempty(currentBridgePoints)
    ii = ii+1;
    currentXYZ = currentBridgePoints(1,:); %start with the first bridge point
    
    xLim = currentXYZ(1)+[0.5,-0.5]*gridSize; %+-50m in x axis from the current point
    yLim = currentXYZ(2)+[0.5,-0.5]*gridSize; %+-50m in y axis from the current point
    
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
    dataSetSkogBridge(:,:,ii) = ptCloud.Location(randomSample,:)';
    returnNumberBlock(:,:,ii) = pointAttributes.LaserReturns(randomSample,:)';
    intensityBlock(:,:,ii) = ptCloud.Intensity(randomSample,:)';
    %classBlock = single(true([1 tileBlockPointNumber 1648]));
   
    % Get point label for bridge point
    pointLabel(:,ii) = pointAttributes.Classification(randomSample,:)'==17;
    
    
end

blockLabel = single(ones([1 ii]));

    dataSetSkogBridge(:,:,(ii+1):end) = [];
    returnNumberBlock(:,:,(ii+1):end) = [];
    intensityBlock(:,:,(ii+1):end) = [];
    pointLabel(:,(ii+1):end) = [];
end

