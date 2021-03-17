function [coordBlock,intensityBlock,returnNumberBlock,pointLabel,blockLabel,blockGeoCoord] = ... 
    generateTileBlocks(fileForCoordinates,generationMethod,gridSize,tileBlockPointNumber,class,varargin)
%generateTileBlocks uses three methods to generate tile blocks from LAZ
% files. The first one only capture tile blocks with bridges and capture
% all bridges in the whole files that are selected. The second method is to
% capture all areas where no tile blocks is present in the whole selected
% files. The third method capture tile blocks within a set of coordinates 
% given in input "fileForCoordinates". It is possible to include
% neighbouring LAZ file in method 3 to get a complite tile block in the
% limits of the LAZ files, it is also possible to deselect all tile blocks
% with bridges.

%   Input: 
%       fileForCoordinates: Contains the file name of the LAZ-files and the
%           selected coordinates within area of each LAZ-file.
%       generationMethod: Should be a array with size "number of
%           LAZ-file"x3, the columns represent which method to generate tile
%           that will be used and rows corresponds to which LAZ file it will be
%           used for. Zeros in the array indicates that the method should 
%           not be used and ones means that the method will be applied.
%       gridSize: Is the side length for every tile block.
%       tileBlockPointNumber: Is the number of points that should be
%           captured in each tile block.
%       class: Indicates which class number that represent bridges.
%   Extra Inputs:
%       'dataPath': Is the path to the LAZ files, if it is not used the
%           function will search for the LAZ-files in the current folder.
%       'neighbours': Will buffer points from neighbouring LAZ-files in
%           method 3, to prevent tile blocks close to limits in the LAZ-files to
%           contain less points.
%   Output:
%       coordBlock: The coordinate for the points in each tile block.
%       intensityBlock: The intensity for the points in each tile block.
%       returnNumberBlock: The return number for the points in each tile block.
%       pointLabel: Indicates if the points have an bridge class in each
%           tile block.
%       blockLabel: Labels all the tile block if they are containing a
%       bridge or not.

    % The index number of each method
    gBridge = 1;
    gNonBridge = 2;
    gCoord = 3;
    % Number of methods.
    nrGenerationMethods = 3;
    
    % Number of required input arguments.
    nrInputs = 5;
    % Number of extra inputs.
    extraInputs = nargin - nrInputs;
    % Number of extra features.
    numberOfFeatures = 2;
    extraFeature = false(numberOfFeatures,1);
    % Set all extra features to false.
    inputNumberFeature = zeros(numberOfFeatures,1);
    
    % Check if there are extra inputs.
    if( extraInputs > 0 )
        for ii=1:2:extraInputs

            % Check which extra features that will be used and get the
            % index of the data inputs.
            if(contains(varargin{ii},"dataPath"))
                extraFeature(1)=true;
                inputNumberFeature(1) = ii+1;
                
            elseif(contains(varargin{ii},"neighbours"))
                extraFeature(2)=true;
                inputNumberFeature(2) = ii+1;
                
            else
                error(['Wrong input argument (',num2str(ii+nrInputs),').']);
            end
        end
    end
    
    % Set parameters for feature 1.
    if(extraFeature(1))
        dataLAZPath = varargin{inputNumberFeature(1)};
    else
        dataLAZPath = [];
    end
    
    % Set parameters for feature 2.
    if(extraFeature(2))
        neighbourFiles = varargin{inputNumberFeature(2)};
    end
    
    % Get the number of LAZ files.
    nrLAZfiles = size(fileForCoordinates,1);
    % Allocate cell array to store point features in tile blocks.    
    bufferPointCoord = cell(nrLAZfiles,nrGenerationMethods);
    bufferPointIntensity = cell(nrLAZfiles,nrGenerationMethods);
    bufferPointReturnNumber = cell(nrLAZfiles,nrGenerationMethods);
    bufferPointClass = cell(nrLAZfiles,nrGenerationMethods);
    bufferTileBlockClass = cell(nrLAZfiles,nrGenerationMethods);
    
    bufferBlockGeoCoord = cell(nrLAZfiles,nrGenerationMethods);
    
    % Generate tile blocks for each LAZ file.
    for ii=1:nrLAZfiles

        % Get file name of the LAZ files and if any selected coordinates 
        % within the index block/LAZ-file.
        coordinates = fileForCoordinates{ii,2};
        LAZfilename = fileForCoordinates{ii,1};
        selectedFile = dir(fullfile(dataLAZPath,['*',LAZfilename]));

        % Check if the current file is available.
        if( ~isempty(selectedFile) )

            % Upload LAZ-file and get classes and point features.
            lasReader = lasFileReader([dataLAZPath,selectedFile.name]);
            [ptCloud,pointAttributes] = readPointCloud(lasReader,'Attributes',["Classification","LaserReturns"]);

            % Check if generation method 1 should be used in this file.
            if(generationMethod(ii,gBridge) == 1)
            
                [bufferPointCoord{ii,gBridge},bufferPointIntensity{ii,gBridge}, ...
                    bufferPointReturnNumber{ii,gBridge}, ...
                    bufferPointClass{ii,gBridge},bufferTileBlockClass{ii,gBridge}, ...
                    bufferBlockGeoCoord{ii,gBridge} ] = ...
                    getBridgeBlock(ptCloud,pointAttributes,class,tileBlockPointNumber,gridSize);
            end
            
            % Check if generation method 2 should be used in this file.
            if(generationMethod(ii,gNonBridge) == 1)
                
                [bufferPointCoord{ii,gNonBridge},bufferPointIntensity{ii,gNonBridge}, ...
                    bufferPointReturnNumber{ii,gNonBridge}, ...
                    bufferPointClass{ii,gNonBridge},bufferTileBlockClass{ii,gNonBridge}, ...
                    bufferBlockGeoCoord{ii,gNonBridge}] = ...
                    getNonbridgeBlock(ptCloud,pointAttributes,class,tileBlockPointNumber,gridSize);
            end
            
            % Check if generation method 3 should be used in this file.
            if(generationMethod(ii,gCoord) == 1)
                % Check if buffering of neighbour index-boxes/LAZ-files is enable.
                if(extraFeature(2))
                    % Get neighbouring LAZ-files.
                    numberOfNeighbours = size(neighbourFiles{ii},1);
                    ptCloudObjects = cell(numberOfNeighbours+1,1);
                    ptAttrObjects = cell(numberOfNeighbours+1,1);

                    % To check if there is any missing neighbour file.
                    cellsWithoutFile = false(numberOfNeighbours+1,1);

                    % Set the current/middle LAZ file first in the arrays
                    % for point features.
                    ptCloudObjects{1} = ptCloud;
                    ptAttrObjects{1} = pointAttributes;

                    % Get the neighbouring laz files.
                    for jj=2:numberOfNeighbours+1
                        % Get the compleate file names of the neighbouring
                        % LAZ-file.
                        tempNeighbourFile = dir(fullfile(dataLAZPath,['*',neighbourFiles{ii}(jj-1,:)]));

                        % If the neighbouring LAZ file is available.
                        if( ~isempty(tempNeighbourFile) )
                            % Read the LAZ-file.
                            tempLasReader = lasFileReader([dataLAZPath,tempNeighbourFile.name]);
                            [tempPTCloud,tempPTAttr] = readPointCloud(tempLasReader,'Attributes',["Classification","LaserReturns"]);
                            % Store the LAZ-file.
                            ptCloudObjects{jj} = tempPTCloud;
                            ptAttrObjects{jj} = tempPTAttr;
                        else
                            % If missing mark it as missing...
                            cellsWithoutFile(jj) = true;
                        end
                    end
                    % Remove all the cells with missing files.
                    ptCloudObjects(cellsWithoutFile==1) = [];
                    ptAttrObjects(cellsWithoutFile==1) = [];
                    
                    % Capture the tile blocks with method 3.
                    [bufferPointCoord{ii,gCoord},bufferPointIntensity{ii,gCoord}, ...
                        bufferPointReturnNumber{ii,gCoord}, ...
                        bufferPointClass{ii,gCoord},bufferTileBlockClass{ii,gCoord}, ...
                        bufferBlockGeoCoord{ii,gCoord} ] = ...
                        getBlockFromCoord(ptCloudObjects,ptAttrObjects,class,tileBlockPointNumber,gridSize, flip(coordinates,2),"neighbours");
                else
                    % Capture the tile blocks with method 3 without neighbour buffering.
                    [bufferPointCoord{ii,gCoord},bufferPointIntensity{ii,gCoord}, ...
                        bufferPointReturnNumber{ii,gCoord}, ...
                        bufferPointClass{ii,gCoord},bufferTileBlockClass{ii,gCoord}, ...
                        bufferBlockGeoCoord{ii,gCoord} ] = ...
                       getBlockFromCoord(ptCloud,pointAttributes,class,tileBlockPointNumber,gridSize, flip(coordinates,2));
                end
            end
        end
    end

    % Make the cells to one dimensional array(merge all the methods used).
    tempCoord = reshape(bufferPointCoord,[nrLAZfiles*nrGenerationMethods,1]);
    tempIntensity = reshape(bufferPointIntensity,[nrLAZfiles*nrGenerationMethods,1]);
    tempReturnNumber = reshape(bufferPointReturnNumber,[nrLAZfiles*nrGenerationMethods,1]);
    tempPointClass = reshape(bufferPointClass,[nrLAZfiles*nrGenerationMethods,1]);
    tempTileBlockClass = reshape(bufferTileBlockClass,[nrLAZfiles*nrGenerationMethods,1]);
    
    tempBlockGeoCoord = reshape(bufferBlockGeoCoord,[nrLAZfiles*nrGenerationMethods,1]);
    

    % Find all the empty cells.
    cellsToRemove = find(cellfun('isempty', tempCoord));
    
    % Remove all the empty cells.
    tempCoord(cellsToRemove) = [];
    tempIntensity(cellsToRemove) = [];
    tempReturnNumber(cellsToRemove) = [];
    tempPointClass(cellsToRemove) = [];
    tempTileBlockClass(cellsToRemove) = [];
    
    tempBlockGeoCoord(cellsToRemove) = [];
    
    % Merge all the tile blocks of different LAZ-files to one array.
    coordBlock = cat(3,tempCoord{:});
    intensityBlock = cat(3,tempIntensity{:});
    returnNumberBlock = cat(3,tempReturnNumber{:});
    pointLabel = cat(2,tempPointClass{:});
    blockLabel = cat(2,tempTileBlockClass{:});
    
    blockGeoCoord = cat(1,tempBlockGeoCoord{:});
    
    blockGeoCoord = flip(blockGeoCoord,2);
    
end

