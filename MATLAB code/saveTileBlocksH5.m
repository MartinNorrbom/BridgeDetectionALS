function saveTileBlocksH5(H5FileName,blockCoord,blockLabel,pointLabel,varargin)
%saveTileBlocksH5 Summary of this function goes here
%   Detailed explanation goes here

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
            
                extraFeature(featureInd) = true;
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
    
    % Create a new h5-file and save slots in the file.
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
        dataNum = int32(1:numberOfBlocks);
        h5write(saveDestination,'/data_num',dataNum );
        
    end
    
    
    % NEED TO IMPROVE THE WAY OF SAVING INTENSITY AND RETURN NUMBER!!!
    if(extraFeature(3) || extraFeature(4))

        h5create(saveDestination,'/normal',[3 tileBlockPointNumber numberOfBlocks],'Datatype','single');

        pointFeatures = single(zeros([3 tileBlockPointNumber numberOfBlocks]));
        
        if(extraFeature(3))
            pointFeatures(1,:,:) = intensityBlock;
        end
        
        if(extraFeature(4))
            pointFeatures(2,:,:) = returnNumberBlock;
        end
        
        h5write(saveDestination,'/normal',pointFeatures );
    end
    
    

%h5create('trainingDataSkog.h5','/indices_split_to_full',[1 tileBlockPointNumber numberOfBlocks],'Chunksize',[1 128 103], 'Deflate',1,'Datatype','int32');

    
end


