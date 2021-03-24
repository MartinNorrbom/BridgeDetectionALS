function [selectedCoordinates,tileBlockGenerationMethod] = ...
    getSelectedCoordinates(CoordinatesPath,sizeIndexBlock,gridSize)
%getSelectedCoordinates search for all the text files with coordinate and
% registrates which operation that are suposed to done for those coordinates
% in the file. The files it check MUST begin with word "Koordinater" in its
% name after that it should contains number from 1 to 3 to indicate which
% operation that will be done on the file. Then it collect all the selected
% coordinates and label the operations that will be done on each coordinate.


%   Input: 
%       CoordinatesPath: The path to the folder where all the text files
%           with coordinates is located. Can also be whole file names.
%       sizeIndexBlock: The side length of the index block/LAZ-files in
%           meters.
%       gridSize: The side length of the tile blocks in meters.
%   Output:
%       selectedCoordinates: An array with all the selected coordinates.
%       tileBlockGenerationMethod: An matrix that indicates which method
%           that will be used to generate tile blocks for each coordinate.

% Here is some examples of valid names for the text files: 
% Example 1: Koordinater2_London.txt
% This file contains coordinates that will only use the second method to
% generate tile blocks.
% Example 2: Koordinater13_Stockholm.txt
% This file contains coordinates that will use first and third method to
% generate tile blocks.
% Example 3: Koordinater123_Karlstad.txt
% This file contains coordinates that will use all the three method to
% generate tile blocks.

    % The start name of the text files that 
    startName = 'Koordinater';
    % Get the length of the start name.
    startLengthName = size(startName,2);

    % The number of methods to generate tile blocks.
    numberOfmehtods = 3;
    
    
    % Check if multiple file is already selected.
    if( isstring(CoordinatesPath) && (length(CoordinatesPath) > 1) )
        % Get the path and the names of multiple files.
        [textFileFolder,textFileNames] = fileparts(CoordinatesPath);
        textFileFolder = textFileFolder(1);
    else
        
        CoordinatesPath = char(CoordinatesPath);
        % Checks if an specific file was selected.
        if( strcmp(CoordinatesPath(end-3:end),'.txt') )

            % Select the specified txt file.
            textFilesCoord = dir(CoordinatesPath);
            
        else
            % Find all the textfiles that starts with the "startName" in 
            % the folder where the files is located.
            textFilesCoord = dir([CoordinatesPath,startName,'*.txt']);
        end
        % Store the path and name in variables.
        textFileFolder = textFilesCoord.folder;
        textFileNames = string({textFilesCoord.name});
    end
    
    % The number of text files with coordinates.
    numberOfFiles = length(textFileNames);
    
    % Allocate matrix to set generation method for each text file.
    generationMethodFile = zeros(numberOfFiles,3);
    
    % Loop through all the names of the text files and get the generation
    % methods that should be used.
    for ii=1:numberOfFiles
        
        % Name of the current text file.
        tempFileName = char(textFileNames(ii));
        % Set generation method for the current text file.
        for jj=1:numberOfmehtods
            % Get a symbol after start name
            tempLetter = tempFileName(startLengthName+jj);
            
            % Check if the symbol is a number.
            if(isstrprop(tempLetter,'digit'))
                % Set generation method.
                generationMethodFile(ii,str2num(tempLetter)) = 1;
                
            else
                % Out of generation methods.
                break;
            end
        end
    end

    % To buffer all the coordinates and methods from each text file.
    bufferCoordinates = cell(numberOfFiles,1);
    bufferMethods = cell(numberOfFiles,1);
    
    % Loop through all the text files to recieve the cordinates.
    for ii=1:numberOfFiles
        % Current text file.
        tempFileName = fullfile(textFileFolder,textFileNames(ii));
        
        % If the text file should use generation method 3 the interpolation
        % of the coordinates needs to be the size of an tile block.
        if(generationMethodFile(ii,3) == 1 )

            bufferCoordinates{ii} = ...
                readCoordinates(tempFileName,gridSize);
    
        elseif( any( generationMethodFile(ii,1:2) == 1 ) )
            % If other methods than method 3 is used, the interpolation can
            % be the half size of one index box/LAZ-file.
    
            bufferCoordinates{ii} = ...
                readCoordinates(tempFileName,sizeIndexBlock/2);
            
        end
        % Store tile block generation method for each coordinate.
        bufferMethods{ii} = repmat(generationMethodFile(ii,:),[size(bufferCoordinates{ii},1),1]);
    
    end
    
    
    % Find all the empty cells.
    filesToRemove = find(cellfun('isempty', bufferCoordinates));
    % Remove the empty cells.
    bufferCoordinates(filesToRemove) = [];
    bufferMethods(filesToRemove,:) = [];
    
    % Return the selectedCoordinates and tileBlockGenerationMethod for
    % those coordinates.
    selectedCoordinates = cell2mat(bufferCoordinates);
    tileBlockGenerationMethod = cell2mat(bufferMethods);
    
end

