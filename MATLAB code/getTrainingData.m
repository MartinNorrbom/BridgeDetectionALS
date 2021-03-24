function getTrainingData(gridSize,tileBlockPointNumber,class,sizeIndexBlock, ...
    H5FileName,generationFolder,coordFiles,statusFile,dataLAZPath, ...
    serverName,path1Server)
%getTrainingData generates tile block from LAZ-files. The purpose is to use
% it as training data for ML algorithms.
%   Input: 
%       gridSize: The side length of each tile block.
%       tileBlockPointNumber: The number of points per tile block.
%       class: The number that represent the disered class in the LAZ-file
%           that should be labeled.
%       sizeIndexBlock: Is the side length of the LAZ-files, based on the
%           assumption that the LAZ-file is shaped like a square.
%       H5FileName: The filename of the generated output data.
%       generationFolder: The location where the data should be generated.
%       coordFiles: The folder (or files) that contains the text file with the
%           coordinates that shapes a polygon to indacate the area where
%           the tile blocks will be generate.
%       statusFile: The status production of the LAZ-files, also contains
%           information about how to find the LAZ-files at the server.
%       dataLAZPath: The path where the LAZ files is located or will be
%           downloaded.
%       serverName: The adress of the server, to download missing
%           LAZ files.
%       path1Server: The path in the server that contains the laser data.


    % --------------- Get the selected coordinates ---------------
    [manualSelectedCoordinates,tileBlockMethod] = ...
        getSelectedCoordinates(coordFiles,sizeIndexBlock,gridSize);

    
    % ----------------- Get the production status -----------------
    dataLocationInfo = shaperead(statusFile);


    % ----- Finds the LAZ-files for the selected coordinates ------
    [filePathsAndNames,fileForCoordinates,generationMethod,neighbourFiles] = ...
        getLAZFileFromCoord(manualSelectedCoordinates, dataLocationInfo,"Methods",tileBlockMethod,"neighbours");

    
    % ------------- Download all the missing files ----------------
    getMissingFilesFromServer(filePathsAndNames,serverName,path1Server,dataLAZPath);
    
    
    % ----- Generate tile-blocks for the selected coordinates -----
    [coordBlock,intensityBlock,returnNumberBlock,pointLabel,blockLabel,blockGeoCoord] = ... 
        generateTileBlocks(fileForCoordinates,generationMethod,gridSize, ...
        tileBlockPointNumber,class,"dataPath",dataLAZPath,"neighbours",neighbourFiles);
    
    % Get the number of tile blocks.
    numberOfBlocks = size(coordBlock,3);

    % Mark each tile block with ID number. (Not required)
    dataNum = int32(1:numberOfBlocks);
    
    % -------- Save the generated tile blocks in h5 format ---------
    saveTileBlocksH5(H5FileName,coordBlock,blockLabel,pointLabel, ...
        "data_num",dataNum,"intensity",intensityBlock, ...
        "returnNumber",returnNumberBlock,"path",generationFolder, ...
        "coordinates",blockGeoCoord');
    
    
end

