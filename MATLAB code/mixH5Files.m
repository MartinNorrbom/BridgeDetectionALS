function mixH5Files(fileName,inputFolder,outputFolder,maxFileSize,proportion,propName)
%mixH5Files Merge h5 files to create data set from different h5 files. When
% this function is running a GUI shows up where the user can select the h5
% files to merge. Then this function check the information that is stored
% in the h5 files and combine the datanames/slots that the files have in
% common. Then it randomize the order of the samples in the data to make it
% ready to be used as input for machine learning. The user can also decide
% the maximum file size of the output files, then multiple output files
% will be generated if the total file size bigger than the maximum file
% size.

%   Input: 
%       fileName: The base name of the output files.
%       inputFolder: The folder where the input data is located.
%       outputFolder: The folder where the output data should be located.
%       maxFileSize: The maximum size of the output files(in megabytes).
%       proportion: Specify the percent of the tile blocks or points that
%           should be labeled positive (1). If the input is empty ("[]"), 
%           all the available data will be combined.
%       propName: Specifies if the "proportion" should be used on tile 
%           blocks labels or segmentation labels. For segmentation the
%           input must be "segmentation" otherwise block labels will be
%           used.

    % Define data types.
    dataTypesToSave = ["double","int32","int32","single","single","single","int32"];
    % Define supported data names/variants in h5.
    dataNameList = ["data","label","label_seg","data_num","geo_coord","intensity","return_number"];
    % Dimensions to merge, while merging data slots from different files.
    catDim = [3,1,2,1,2,3,3];
    % Get number of data slots.
    numberOfDataSlots = length(dataNameList);

    % Filter to find files faster.
    selectFilter = [inputFolder,'*.h5'];
    % UI to find h5 files.
    [file,path] = uigetfile( selectFilter,'MultiSelect','on');
    
    % To count the total size of all files.
    sizeOfFiles = 0;
    
    % Check if one or multiple files was selected.
    if(iscell(file))
        % Multiple files was selected.
        nrOfFiles = length(file);
        
        % Variable to store the available slots in all files.
        namesAvailable = zeros(nrOfFiles,numberOfDataSlots);
        
        % Loop through all the files.
        for ii=1:nrOfFiles
            
            % Get the size of the current file and store it.
            selectedFiles = dir([path,file{ii}]);
            sizeOfFiles = sizeOfFiles + selectedFiles.bytes;

            % Get the datanames/slot-names in the current file.
            h5InfoStruct = h5info([path,file{ii}]);
            dataNames = {h5InfoStruct.Datasets.Name};
            
            % Check which slots that are available of them in the list.
            for jj=1:numberOfDataSlots
                for kk=1:length(dataNames)
                    if(strcmp(dataNames{kk},dataNameList(jj)))
                        namesAvailable(ii,jj) = 1;
                        break;
                    end
                end
            end

        end
        
    else
        % There were only one file selected.
        nrOfFiles = 1;
        % Get the size of the file.
        selectedFiles = dir([path,file]);
        sizeOfFiles =  selectedFiles.bytes;
        
        % Store the datanames/slot-names.
        namesAvailable = zeros(nrOfFiles,numberOfDataSlots);
        
        % Get the datanames/slot-names in the current file.
        h5InfoStruct = h5info([path,file]);
        dataNames = {h5InfoStruct.Datasets.Name};
        
        % Check which slots that are available of them in the list.
        for jj=1:numberOfDataSlots
            for kk=1:length(dataNames)
                if(strcmp(dataNames{kk},dataNameList(jj)))
                    namesAvailable(1,jj) = 1;
                    break;
                end
            end
        end
        % Make this varaible into a cell.
        file = {file};
    end
    
    % Get all the slots that the selected files has in common and only
    % collect and store that kind of data.
    namesIncluded = (sum(namesAvailable,1) == nrOfFiles);
    
    % Allocate space to read the data slots from the h5-file.
    dataFromFiles = cell(nrOfFiles,numberOfDataSlots);
    
    % Loop through all h5-files.
    for ii=1:nrOfFiles
        % Get the location of the current h5-file.
        tempFileName = fullfile(path,file{ii});
        
        % Loop through all the data slots.
        for jj=1:numberOfDataSlots
            % Save data if slot is valid.
            if(namesIncluded(jj))
                dataFromFiles{ii,jj} = h5read(tempFileName,strcat("/",dataNameList(jj)));
            end
        end
    end

 
    % Allocate variable to store data merge from different files.
    dataSlots = cell(numberOfDataSlots,1);
    
    % Loop through all the data names/slots.
    for ii=1:numberOfDataSlots
        % Checks if slot is valid.
        if(namesIncluded(ii))
            % Merge the same data slot from different files.
            slotFiles = dataFromFiles(:,ii);
            dataSlots{ii} = cat(catDim(ii),slotFiles{:});
        end
    end
    
    % Check if proportion object is specified.
    if(~isempty(propName))
        % Check proportion name and if segmentation label is provided in the data.
        if(strcmp(propName,"segmentation") && namesIncluded(3))
        
            disp(strcat("The proportion of segmentation labels is: ",num2str(proportion)))
            
            % Get segmentation label.
            allLabel_Seg = dataSlots{3};
            
            % Get number of blocks.
            nrSamples = size(allLabel_Seg,2);
            % Get number of points.
            nrPoints = size(allLabel_Seg,1);
            % Get number of positive and negative labels of segmentation
            % label.
            nrN_Seg = sum(allLabel_Seg == 0,'all');
            nrP_Seg = sum(allLabel_Seg == 1,'all');
            
            % Get the number of negative labels to not change the number of
            % positive labels.
            NrN_Seg_Req = nrP_Seg/proportion-nrP_Seg;
            % Get the number of positive labels to not change the number of
            % negative labels.
            NrP_Seg_Req = nrN_Seg/(1-proportion)-nrN_Seg;
            
            % Get the labels of the samples (could have read dataSlots{2}... )
            Plabel_sIndex = sum(allLabel_Seg,1) >= 1;
            Plabel_sample = sum( Plabel_sIndex );
            Nlabel_sample = sum( sum(allLabel_Seg,1) == 0 );
            
            % Check if it is possible to change the number of negative
            % segmentation labels to fulfill the desired proportion.
            if( NrN_Seg_Req <= nrN_Seg )
                
                NrN_Seg_With_Bridge_Label = sum(allLabel_Seg(:,Plabel_sIndex)==0,'all');
                
                % Get the number of negative samples needed. 
                reducedN = round( (NrN_Seg_Req-NrN_Seg_With_Bridge_Label)/nrPoints );
                
                % Check if proportion is possible.
                if( reducedN < 0 )
                    % Not possible try to do as best it can bee.
                    disp("Can not upfill label segmentation proportion.")
                    
                    nrN_Seg = nrN_Seg - Nlabel_sample*nrPoints;
                    
                    disp(strcat("The generated proportion will be: ",num2str(nrP_Seg/(nrN_Seg+nrP_Seg))))
                    proportion = 1;
                    
                else
                    % Possible to establish the proportion.
                    proportion = Plabel_sample/(Plabel_sample+reducedN);
                end
                
            % Check if it is possible to change the number of positive
            % segmentation labels to fulfill the desired proportion.
            elseif( NrP_Seg_Req <= nrP_Seg )
                
                % Get the average number of positive segment label per
                % positive sample.
                poistivePointsPerSample = nrP_Seg/Plabel_sample;
                
                % Get the number of positive samples needed.
                reducedP = round( NrP_Seg_Req/poistivePointsPerSample );
                
                % Get the proportion
                proportion = reducedP/(reducedP+Nlabel_sample);

            else
                disp("WARNING: The specified propotion of segmentation labels is not possible to generate.")
                error("Error: Combine data failed.")     
            end
        
        end
    end
    
    
    % Check if the data set should be balanced and that the data is labeled.
    if(~isempty(proportion) && namesIncluded(2))
        % Print the desired proportion of the data.
        disp(strcat("The proportion of labels is: ",num2str(proportion)))
        % Get all labels.
        allLabels = dataSlots{2};
        % Get indecies of positive and negative labels.
        indPLabels = find( allLabels == 1 );
        indNLabels = find( allLabels == 0 );
        
        % Get an array with all indecies of the labels.
        allIndex = 1:length(allLabels);
        % Get number of positive and negaive labels.
        nrPL = length(indPLabels);
        nrNL = length(indNLabels);
        
        % Get desired number of negative labels to keep positive labels
        % untouched.
        PDesiredNrN = nrPL/proportion-nrPL;
        % Get desired number of positive labels to keep negative labels
        % untouched.
        NDesiredNrP = nrNL/(1-proportion)-nrNL;
        
        % If negaitve labels can be reduced.
        if(PDesiredNrN <= nrNL)
            
            % Get a set of negative indecies to remove.
            indexToRm = randperm(nrNL, (nrNL - ceil(PDesiredNrN)));
            % Get global indecies to remove of the negative indecies.
            indexToRm = indNLabels(indexToRm);
            
        % If positive labels can be reduced.
        elseif(NDesiredNrP <= nrPL)
        
            % Get a set of positive indecies to remove.
            indexToRm = randperm(nrPL, (nrPL - ceil(NDesiredNrP)));
            % Get global indecies to remove of the positive indecies.
            indexToRm = indPLabels(indexToRm);
            
        else
            
            % If the requirements can not be satisfied.
            disp("WARNING: The specified propotion is not possible to generate.")
            error("Error: Combine data failed.")            
        end
        
        % Remove gloabel indecies to fulfill the proportion.
        allIndex(indexToRm) = [];
        
        % Get number of samples.
        numberOfBlocks = length(allIndex);
        
        % Get a random sequence of indicies.
        randomSequence = randperm(numberOfBlocks);
        % Randomize the data.
        randomIndex = allIndex(randomSequence);
    
    % No proportion is required.
    else
        
        % Get number of tile-blocks/samples in the data.
        numberOfBlocks = size(dataSlots{1},3);
        
        % Randomize the data
        randomIndex = randperm(numberOfBlocks);
        
    end
    

    % Get the number of points per tile-block/sample.
    tileBlockPointNumber = size(dataSlots{1},2);
    
    % To get size in Megabytes
    sizeOfFiles = round(sizeOfFiles/10^6);
    % Get the number of output files that is going to be created.
    numberOutputFiles = ceil( (numberOfBlocks/size(dataSlots{1},3))*sizeOfFiles/maxFileSize);
    
    % Indexes of tile-blocks for each file.
    blockIndexFiles = ceil(linspace(0,numberOfBlocks,numberOutputFiles+1));
    
    % Get path location and name of the file that will be generated.
    saveDestination = [outputFolder,fileName];
    
    % To represent which number the file is in the filename.
    fileNumbers = num2str((1:numberOutputFiles)');
    
    % Loop through all output files.
    for ii=1:numberOutputFiles
        
        % Get the save name and path in one string.
        saveName = [saveDestination(1:end-3),'_',fileNumbers(ii,:),saveDestination(end-2:end)];
        
        % Delete if there already exist a file with the same name.
        delete(saveName);
        
        % Get the current indexies of the tile block.
        lowIndex = (blockIndexFiles(ii)+1);
        highIndex = blockIndexFiles(ii+1);
        
        % Get the number of blocks in the current file.
        numberOfBlocks = highIndex-blockIndexFiles(ii);
        
        % Get the current set of random indecies.
        currentIndex = randomIndex(lowIndex:highIndex);
        
        % Save all data in the data slots of the created h5-file.
        if(namesIncluded(1))
            h5create(saveName,strcat("/",dataNameList(1)),[size(dataSlots{1},1) tileBlockPointNumber numberOfBlocks],'Datatype', dataTypesToSave(1));
            
            h5write(saveName,strcat("/",dataNameList(1)),dataSlots{1}(:,:,currentIndex) );
        end
        
        if(namesIncluded(2))
            h5create(saveName,strcat("/",dataNameList(2)),numberOfBlocks, 'Datatype',dataTypesToSave(2));
            
            h5write(saveName,strcat("/",dataNameList(2)),dataSlots{2}(currentIndex) );
        end
        
        
        if(namesIncluded(3))
            h5create(saveName,strcat("/",dataNameList(3)),[tileBlockPointNumber numberOfBlocks], 'Datatype',dataTypesToSave(3));
            
            h5write(saveName,strcat("/",dataNameList(3)),dataSlots{3}(:,currentIndex) );
        end
        
        
        if(namesIncluded(4))
            h5create(saveName,strcat("/",dataNameList(4)),numberOfBlocks, 'Datatype',dataTypesToSave(4));
            
            h5write(saveName,strcat("/",dataNameList(4)),dataSlots{4}(currentIndex) );
        end
        

        if(namesIncluded(5))
            h5create(saveName,strcat("/",dataNameList(5)),[2 numberOfBlocks], 'Datatype',dataTypesToSave(5));
            
            h5write(saveName,strcat("/",dataNameList(5)),dataSlots{5}(:,currentIndex) );
        end
        
        
        if(namesIncluded(6))
            h5create(saveName,strcat("/",dataNameList(6)),[1 tileBlockPointNumber numberOfBlocks], 'Datatype',dataTypesToSave(6));
            
            h5write(saveName,strcat("/",dataNameList(6)),dataSlots{6}(1,:,currentIndex) );
        end
        
                
        if(namesIncluded(7))
            h5create(saveName,strcat("/",dataNameList(7)),[1 tileBlockPointNumber numberOfBlocks], 'Datatype',dataTypesToSave(7));
            
            h5write(saveName,strcat("/",dataNameList(7)),dataSlots{7}(1,:,currentIndex) );
        end
        
    end
    
end

