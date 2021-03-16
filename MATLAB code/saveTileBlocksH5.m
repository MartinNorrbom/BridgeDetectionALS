function saveTileBlocksH5(H5FileName,blockCoord,blockLabel,pointLabel,varargin)
%saveTileBlocksH5 Saves the generated tile blocks to h5-format, so it can
% be used as input for machine learning algorithms such as poinNet and
% pointCNN. If no path is specified the h5-files will be save in the same
% folder where the program is located. 

%   Input: 
%       H5FileName: Is the file name that will be saved, it must contain
%           ".h5" at the end.
%       blockCoord: Contains the coordinates of the points in the tile
%           blocks.
%       blockLabel: Labels each tile block.
%       pointLabel: Labels each point in the tile blocks.
%   Extra Inputs:
%       'path': Is the path for the location where the h5 is suposed to be
%           saved.
%       'data_num': Is the index name for each tile block.
%       'intensity': Is to save the intensity of each point.
%       'returnNumber': Is to save the return number of each point.
%       'coordinates': The location of each tile-block in geographic
%           coordinates.

    % Check if there are any extra inputs.
    nrReqInputs = 4;
    extraInputs = nargin-nrReqInputs;
    
    nrExFeatures = 5;
    
    % To store information of extra inputs.
    extraFeature = false(nrExFeatures,1);
    extraFeatureInd = zeros(nrExFeatures,1);
    
    
    % Define data types.
    dataTypesToSave = ["double","int32","int32","single","single","single","int32"];
    % Define supported data names/variants in h5.
    dataNameList = ["data","label","label_seg","data_num","geo_coord","intensity","return_number"];
    
    
    % Names of the extra inputs.
    featureNamesWithInputArgument = ["path","data_num","intensity","returnNumber","coordinates"];

    % Check if there are any extra inputs.
    if( extraInputs > 0 )
        
        ii = 1;
        while( ii <= extraInputs )

            % Check which extra features that will be used and get the
            % index of the data inputs.
            
            featureInd = find( contains(featureNamesWithInputArgument,varargin{ii}) );
            

            if(~isempty(featureInd) && (length(featureInd) <= 1))
                % Save basic info of the input.
                extraFeature(featureInd) = true;
                extraFeatureInd(featureInd) = ii+1;
                ii = ii+2;
            else
                error(['Wrong input argument (',num2str(ii+nrReqInputs),').']);
            end
            
        end
    end

    % Check where to save the file.
    if(extraFeature(1))
        saveDestination = [varargin{extraFeatureInd(1)},H5FileName];
    else
        saveDestination = H5FileName;
    end

    
    
    % Get parameters of tile blocks.
    tileBlockPointNumber = size(blockCoord,2);
    numberOfBlocks = size(blockCoord,3);
    
    % Create a new h5-file and save slots in the file. Delete if there
    % already exist an h5-file with the same name.
    delete(saveDestination);
    h5create(saveDestination,strcat("/",dataNameList(1)),[size(blockCoord,1) tileBlockPointNumber numberOfBlocks],'Datatype', dataTypesToSave(1));
    h5create(saveDestination,strcat("/",dataNameList(2)),numberOfBlocks, 'Datatype',dataTypesToSave(2));
    h5create(saveDestination,strcat("/",dataNameList(3)),[tileBlockPointNumber numberOfBlocks], 'Datatype',dataTypesToSave(3));


    % Save the tile blocks standard data to h5 format.
    h5write(saveDestination,strcat("/",dataNameList(1)),blockCoord );
    h5write(saveDestination,strcat("/",dataNameList(2)),blockLabel );
    h5write(saveDestination,strcat("/",dataNameList(3)),pointLabel );
    

    % Save extra features.
    if(extraFeature(2))

        h5create(saveDestination,strcat("/",dataNameList(4)),numberOfBlocks, 'Datatype','int32');
        h5write(saveDestination,strcat("/",dataNameList(4)),varargin{extraFeatureInd(2)} );
        
    end
    
    if(extraFeature(5))
        h5create(saveDestination,strcat("/",dataNameList(5)),[2 numberOfBlocks], 'Datatype',dataTypesToSave(5));

        h5write(saveDestination,strcat("/",dataNameList(5)),varargin{extraFeatureInd(5)} );
    end


    if(extraFeature(3))
        h5create(saveDestination,strcat("/",dataNameList(6)),[tileBlockPointNumber numberOfBlocks], 'Datatype',dataTypesToSave(6));

        h5write(saveDestination,strcat("/",dataNameList(6)),varargin{extraFeatureInd(3)} );
    end


    if(extraFeature(4))
        h5create(saveDestination,strcat("/",dataNameList(7)),[tileBlockPointNumber numberOfBlocks], 'Datatype',dataTypesToSave(7));

        h5write(saveDestination,strcat("/",dataNameList(7)),varargin{extraFeatureInd(4)} );
    end
    
end


