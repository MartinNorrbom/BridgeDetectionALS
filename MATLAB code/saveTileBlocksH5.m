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

    % Check if there are any extra inputs.
    nrReqInputs = 4;
    extraInputs = nargin-nrReqInputs;
    
    % To store information of extra inputs.
    extraFeature = false(4,1);
    extraFeatureInd = zeros(4,1);
    
    % Names of the extra inputs.
    featureNames = ["path","data_num","intensity","returnNumber"];

    % Check if there are any extra inputs.
    if( extraInputs > 0 )
        
        ii = 1;
        while( ii <= extraInputs )

            % Check which extra features that will be used and get the
            % index of the data inputs.
            
            featureInd = find( contains(featureNames,varargin{ii}) );
            
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
    h5create(saveDestination,'/data',[size(blockCoord,1) tileBlockPointNumber numberOfBlocks],'Datatype', 'double');
    h5create(saveDestination,'/label',numberOfBlocks, 'Datatype','int32');
    h5create(saveDestination,'/label_seg',[tileBlockPointNumber numberOfBlocks], 'Datatype','int32');


    % Save the tile blocks standard data to h5 format.
    h5write(saveDestination,'/data',blockCoord );
    h5write(saveDestination,'/label',blockLabel );
    h5write(saveDestination,'/label_seg',pointLabel );
    

    % If data_num needs to be saved.
    if(extraFeature(2))

        h5create(saveDestination,'/data_num',numberOfBlocks, 'Datatype','int32');
        h5write(saveDestination,'/data_num',varargin{extraFeatureInd(2)} );
        
    end
    
    
    % NEED TO IMPROVE THE WAY OF SAVING INTENSITY AND RETURN NUMBER!!!
    
    % If any extra feature is included.
    if(extraFeature(3) || extraFeature(4))

        % Create a slot to store extra features from the points in the tile
        % blocks.
        h5create(saveDestination,'/normal',[3 tileBlockPointNumber numberOfBlocks],'Datatype','single');

        pointFeatures = single(zeros([3 tileBlockPointNumber numberOfBlocks]));
        
        % If intensity is included it will be saved in extra feature.
        if(extraFeature(3))
            pointFeatures(1,:,:) = varargin{extraFeatureInd(3)};
        end
        
        % If return number is included it will be saved in extra feature.
        if(extraFeature(4))
            pointFeatures(2,:,:) = varargin{extraFeatureInd(4)};
        end
        
        % Save the extra features.
        h5write(saveDestination,'/normal',pointFeatures );
    end
    
    

%h5create('trainingDataSkog.h5','/indices_split_to_full',[1 tileBlockPointNumber numberOfBlocks],'Chunksize',[1 128 103], 'Deflate',1,'Datatype','int32');

    
end


