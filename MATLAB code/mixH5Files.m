function [sizeOfFiles] = mixH5Files(fileName,inputFolder,outputFolder,maxFileSize)
%mixH5Files Summary of this function goes here
%   Detailed explanation goes here


    % Define data types.
    dataTypesToSave = ["double","int32","int32","single","single","single","int32"];
    % Define supported data names/variants in h5.
    dataNameList = ["data","label","label_seg","data_num","geo_coord","intensity","return_number"];
    % Dimensions to merge, while merging data slots from different files.
    catDim = [3,1,2,1,1,2,2];
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
    
    % Get number of tile-blocks/samples in the data.
    numberOfBlocks = size(dataSlots{1},3);
    % Get the number of points per tile-block/sample.
    tileBlockPointNumber = size(dataSlots{1},2);
    
    % Randomize the data
    randomIndex = randperm(numberOfBlocks);
    
    % To get size in Megabytes
    sizeOfFiles = round(sizeOfFiles/10^6);
    % Get the number of output files that is going to be created.
    numberOutputFiles = ceil(sizeOfFiles/maxFileSize);
    
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
            h5create(saveName,strcat("/",dataNameList(6)),[tileBlockPointNumber numberOfBlocks], 'Datatype',dataTypesToSave(6));
            
            h5write(saveName,strcat("/",dataNameList(6)),dataSlots{6}(:,currentIndex) );
        end
        
                
        if(namesIncluded(7))
            h5create(saveName,strcat("/",dataNameList(7)),[tileBlockPointNumber numberOfBlocks], 'Datatype',dataTypesToSave(7));
            
            h5write(saveName,strcat("/",dataNameList(7)),dataSlots{7}(:,currentIndex) );
        end
        
    end
    
end

