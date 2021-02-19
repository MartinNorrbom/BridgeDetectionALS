function [dataSetSkogNonbridge,returnNumberBlock,intensityBlock,pointLabel,blockLabel] = getNonbridgeBlock(ptCloud,pointAttributes,class,tileBlockPointNumber,gridSize)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
x = ptCloud.XLimits(1):gridSize:ptCloud.XLimits(2);
y = ptCloud.YLimits(1):gridSize:ptCloud.YLimits(2);
[X,Y] = meshgrid(x,y);

numberOfBlock = (length(x)-1)*(length(y)-1);
% numberOfBlock = (length(x)-1)*(ii-1)+jj;
dataSetSkogNonbridge = single(zeros([3 tileBlockPointNumber numberOfBlock]));
returnNumberBlock = single(zeros([1 tileBlockPointNumber numberOfBlock]));
intensityBlock = single(zeros([1 tileBlockPointNumber numberOfBlock]));

tileBlockClass = false([numberOfBlock,1]);
% bridgePoints = pointAttributes.Classification == class;
% 
% xyzBridgePoints = ptCloud.Location(bridgePoints,:);

currentPoints = ptCloud.Location; 

    for ii=1:(length(x)-1)
        
        for jj = 1: (length(y)-1)
            % Find points in the block limit
             pointsInBlockLimit = ...
            (currentPoints(:,1)<=X(ii,jj+1)) & (currentPoints(:,1)> X(ii,jj))& ...
            (currentPoints(:,2)<=Y(ii+1,jj)) & (currentPoints(:,2)>Y(ii,jj));

         % Remove selected bridge points from currentPoints.
            %currentPoints(pointsInBlockLimit,:) = [];    

        %  xyzPointsInBlockLimit =  ptCloud.Location(pointsInBlockLimit,:);
            numberOfClass = pointAttributes.Classification(find(pointsInBlockLimit),:);

            if ~any(numberOfClass(:)==class)

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

                    %dataSetSkogNonbridge(:,:,((length(x)-1)*(ii-1)+jj)) = ptCloud.Location(randomSample,:)';
                
                    dataSetSkogNonbridge(:,:,((length(x)-1)*(ii-1)+jj)) = ptCloud.Location(randomSample,:)';
                    returnNumberBlock(:,:,ii) = pointAttributes.LaserReturns(randomSample,:)';
                    intensityBlock(:,:,ii) = ptCloud.Intensity(randomSample,:)';
                    
            else
                tileBlockClass( ((length(x)-1)*(ii-1)+jj) ) = true;
            end
        
        end
    end 
    
    dataSetSkogNonbridge(:,:,tileBlockClass) = [];
    returnNumberBlock(:,:,tileBlockClass) = [];
    intensityBlock(:,:,tileBlockClass) = [];
    
    pointLabel = single(false([tileBlockPointNumber numberOfBlock-length(find(tileBlockClass))]));
    blockLabel = single(zeros([1 numberOfBlock-length(find(tileBlockClass))]));
    
    

    
end

