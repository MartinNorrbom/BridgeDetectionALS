function [dataSetSkog,returnNumberBlock,intensityBlock,pointLabel,blockLabel] = ...
    getBlockFromCoord(ptCloud,pointAttributes,class,tileBlockPointNumber,gridSize, coordinates)
%getBlockFromCoord Summary of this function goes here
%   Detailed explanation goes here

    numberOfBlock = size(coordinates,1);

    coordinateCheckList = false(numberOfBlock,1);
    
    % Allocate space to return tile block data.
    dataSetSkog = single(zeros([3 tileBlockPointNumber numberOfBlock]));
    returnNumberBlock = single(zeros([1 tileBlockPointNumber numberOfBlock]));
    intensityBlock = single(zeros([1 tileBlockPointNumber numberOfBlock]));
    pointLabel = single(zeros([tileBlockPointNumber numberOfBlock]));
    blockLabel = single(ones([1 numberOfBlock]));

    ii=0;
    coordCount = 0;
    while sum(coordinateCheckList) < numberOfBlock
        coordCount = coordCount+1;
        if(coordinateCheckList(coordCount) == false)
            ii = ii+1;

            xLim = coordinates(coordCount,1)+[1,-1]*gridSize; %+-50m in x axis from the current point
            yLim = coordinates(coordCount,2)+[1,-1]*gridSize; %+-50m in y axis from the current point

            % Select the coordinates that lie inside the tile block
            coordInBlockLimit = ...
            (coordinates(:,1)<=xLim(1)) & (coordinates(:,1)>xLim(2))& ...
            (coordinates(:,2)<=yLim(1)) & (coordinates(:,2)>yLim(2));

            % remove coordinates that are within the tile block.
            coordinateCheckList(coordInBlockLimit) = true; 

            % Select all points that lie inside the tile block
            pointsInBlockLimit = ...
            (ptCloud.Location(:,1)<=xLim(1)) & (ptCloud.Location(:,1)>xLim(2))& ...
            (ptCloud.Location(:,2)<=yLim(1)) & (ptCloud.Location(:,2)>yLim(2));


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
        blockLabel(1,(ii+1):end) = [];
        dataSetSkog(:,:,(ii+1):end) = [];
        returnNumberBlock(:,:,(ii+1):end) = [];
        intensityBlock(:,:,(ii+1):end) = [];
        pointLabel(:,(ii+1):end) = [];
    end

end

