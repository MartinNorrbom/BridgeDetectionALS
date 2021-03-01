function [selectedCoordinates,coordinatesFound] = ...
    readCoordinates(CoordinatesPath,CoordinatesFileName,coordDistance)
%readCoordinates Groups and interpolated coordinates in a text-file. Empty
% spaces between coordninates futherst down in the file indicates that 
% there is multple groups of coordinate that forms a polygon.
%   Input: 
%       CoordinatesPath: The path from to the text-file with coordinates.
%       CoordinatesFileName: The file name of text-file thatcontains the
%           coordinates.
%       coordDistance: The distance between the interpolated coordinates.
%   Output:
%       selectedCoordinates: A list of the new coordinates.


    % This function is adapted to the text file generated from the website
    % havochvatten.
    % To get the "SWEREF 99 TM" coordinates, downloaded the coordinates from:
    % https://karthavet.havochvatten.se/visakoordinater/


    % Read the text file for the selected coordinates.
    textDataSelectedPoints = readlines([CoordinatesPath,CoordinatesFileName]);
    
    
    % Get the "SWEREF 99 TM" coordinates
    coordRef = "Meterkoordinater i SWEREF99 TM:";
    % To get the group of coordinates.
    areaRef = "* Originalrader *";

    % Create empty variables to find string matches.
    CoordRefMatch = [];
    areaRefMatch = [];
    
    % Get the number of rows in the list.
    numberOfRows = length(textDataSelectedPoints);
    
    % To find which rows that contains the coordinates.
    for ii=1:numberOfRows
        % Change index untill the correct text row is find. 
        if(isempty(CoordRefMatch))
            CoordRefMatch = strfind( textDataSelectedPoints(ii), coordRef );
            startRefIndex = ii;
        end
        
        % Change index untill the correct text row is find.
        if(isempty(areaRefMatch))
            areaRefMatch = strfind(textDataSelectedPoints(ii),areaRef);
            areaRefIndex = ii;
        end

    end
    
    % Get all the "SWEREF 99 TM" coordinates in the textfile.
    endRefIndex = startRefIndex;
    while( contains( textDataSelectedPoints(endRefIndex+1),"N") && ...
            ( (endRefIndex+startRefIndex+1) <= numberOfRows ) )
        % Increase the row index until there are no more coordinates.
        endRefIndex = endRefIndex+1;
    end
    
    % The number of cordinates.
    numberOfCoord = endRefIndex-startRefIndex;
    
    % Convert the text data to a char array.
    SWEREFcoord = char( textDataSelectedPoints((startRefIndex+1):(endRefIndex)) );
    
    % Get the north and east coordinates as doubles.
    northing = str2double(string(SWEREFcoord(:,3:9)));
    easting = str2double(string(SWEREFcoord(:,13:18)));
    
    
    kk = areaRefIndex+1;
    currentCoord = 1;
    currentGroupNumber = 1;
    
    % Group all the coordinates that forms a polygon. 
    coordinateGroup = zeros(numberOfCoord,1);
    while( any(coordinateGroup == 0) )
        % If "N" is present it means that there are coordinates in the row.
        if( contains( textDataSelectedPoints(kk),"N") )
            % Indicate which group the coordinate belongs to.
            coordinateGroup(currentCoord) = currentGroupNumber;
            currentCoord = currentCoord+1;
        else
            % If there is a empty row it indiates that there is a new group comming. 
            currentGroupNumber = currentGroupNumber + 1;
        end
        % Move to a new line in the textfile.
        kk = kk+1;
    end
    
    % Return interpolated coordinates.
    selectedCoordinates = interPolateCoordInGroup([northing,easting],coordinateGroup,coordDistance);

    % Return found coordinates in the text file and the grouping.
    coordinatesFound = [northing,easting,coordinateGroup];
    
end


function [interpolatedCoord] = interPolateCoordInGroup(coordinates,coordinateGroup,coordDistance)
%interPolateCoordInGroup interpolate groups of coordinates to archive a
% higher resolution of the coordinates. The shape of the output is a polygon
% of the input.
%   Input: 
%       coordinates: A list of 2d coordinates. It is required that the
%           points is sorted as the polygon.
%       coordinateGroup: An array with that indicate which group each
%           coordinates belong to.
%       coordDistance: The distance between the interpolated coordinates.
%   Output:
%       interpolatedCoord: A list of the new coordinates.


    % Get polygon/group info.
    uniqueValues = unique(coordinateGroup);
    numberOfGroups = length(uniqueValues);

    % Allocate cells to store interpolated coordinates.
    groupCoord = cell(numberOfGroups,1);
    
    % Loop through all groups of coordinate to make interpolation of the
    % polygon.
    for ii=1:numberOfGroups

        % Get the current group/polygon of coordinates.
        indexforGroup = uniqueValues(ii) == coordinateGroup;
        currentGroup = coordinates(indexforGroup,:);
        
        % Interpolate coordinates if it is more than two.
        if sum(indexforGroup) > 2

            % Get max and min values of the polygon.
            minN = min(currentGroup(:,1));
            maxN = max(currentGroup(:,1));
            minE = min(currentGroup(:,2));
            maxE = max(currentGroup(:,2));

            % Create vetcors to make an temporary square of coordinates.
            intNcoords = (minN:coordDistance:maxN)';
            intEcoords = (minE:coordDistance:maxE)';
            
            % Create pairs of coordinate within the square.
            repNcoord = repmat(intNcoords,length(intEcoords),1);
            repEcoord = repelem(intEcoords,length(intNcoords));

            % Get the indexes for the coordinates of the square to indicate
            % which coordinates that are within the polygon.
            in = inpolygon(repNcoord,repEcoord,coordinates(indexforGroup,1),coordinates(indexforGroup,2));
        
            % Store the coordinates that are within the polygon.
            groupCoord{ii} = [repNcoord(in),repEcoord(in)];
        else
            % If there is less than two coordinate interpolation is sciped
            groupCoord{ii} = currentGroup;
        end
        
        % To check if the code works
        % scatter(repEcoord(in),repNcoord(in));
    end
    
    % Return all the interpolated coordinates within the polygon.
    interpolatedCoord = cell2mat(groupCoord);
    
end