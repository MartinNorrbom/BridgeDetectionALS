function [sizeOfFiles] = mixH5Files(fileName,inputFolder,outputFolder,maxFileSize)
%mixH5Files Summary of this function goes here
%   Detailed explanation goes here

    selectFilter = [inputFolder,'*.h5'];

    [file,path] = uigetfile( selectFilter,'MultiSelect','on');
    
    sizeOfFiles = 0;
    
    if(iscell(file))
        nrOfFiles = length(file);
<<<<<<< Updated upstream
=======
        dataNames = cell(nrOfFiles,1);
>>>>>>> Stashed changes
        
        for ii=1:nrOfFiles
            selectedFiles = dir([path,file{ii}]);
            sizeOfFiles = sizeOfFiles + selectedFiles.bytes;
<<<<<<< Updated upstream
=======
            
            % Store the datanames/slot-names.
            h5InfoStruct = h5info([path,file{ii}]);
            dataNames{ii} = {h5InfoStruct.Datasets.Name};
>>>>>>> Stashed changes
        end
        
    else
        nrOfFiles = 1;
        selectedFiles = dir([path,file]);
        sizeOfFiles =  selectedFiles.bytes;
        
<<<<<<< Updated upstream
=======
        
        % Store the datanames/slot-names.
        h5InfoStruct = h5info([path,file]);
        dataNames = {h5InfoStruct.Datasets.Name};
        
>>>>>>> Stashed changes
        file = {file};
    end
    
    
<<<<<<< Updated upstream

    
    dataTypesToSave = ["double","int32","int32","single"];
    
    
    pointCoordC = cell(nrOfFiles,1);
    blockLabelC = cell(nrOfFiles,1);
    pointLabelC = cell(nrOfFiles,1);
    %pointFeatC = cell(nrOfFiles,1);
    pointDataNumC = cell(nrOfFiles,1);
    
    for ii=1:nrOfFiles
        tempFileName = fullfile(path,file{ii});
        pointCoordC{ii} = h5read(tempFileName,'/data');
        blockLabelC{ii} = h5read(tempFileName,'/label');
        pointLabelC{ii} = h5read(tempFileName,'/label_seg');
        
        pointDataNumC{ii} = h5read(tempFileName,'/data_num' );
        %pointFeatC{ii} = h5read(tempFileName,'/normal');
        
    end

    pointCoord = cat(3,pointCoordC{:});
    blockLabel = cat(1,blockLabelC{:});
    pointLabel = cat(2,pointLabelC{:});
    pointDataNum = cat(1,pointDataNumC{:});
    %pointFeat = cat(3,pointFeatC{:});
=======
    

    % Define data types.
    dataTypesToSave = ["double","int32","int32","single","single","single","int32"];
    dataNameList = ["data","label","label_seg","data_num","geo_coord","intensity","return_number"];
    namesIncluded = false(length(dataNameList),1);
    numberOfDataSlots = length(namesIncluded);
    
    
    % Get the data names that all the files got in common.
    
    dataFromFiles = cell(nrOfFiles,numberOfDataSlots);
    
    for ii=1:nrOfFiles
        
        tempFileName = fullfile(path,file{ii});
        
        
        % Loop through all the selected dataNames/slots in the current file.
        for jj=1:numberOfDataSlots
            % Check if dataName/slot is included.
            if(namesIncluded(jj))
                % Store the data.
                dataFromFiles{ii,jj} = h5read(tempFileName,strcat("/",dataNameList(jj)));
            end
        end
        
    end

    pointCoord = cat(1,dataFromFiles{:});
>>>>>>> Stashed changes
    
    numberOfBlocks = size(pointCoord,3);
    tileBlockPointNumber = size(pointCoord,2);
    
    % Randomize the data
    randomIndex = randperm(numberOfBlocks);
    
    pointCoord(:,:,randomIndex) = pointCoord;
    blockLabel(randomIndex) = blockLabel;
    pointLabel(:,randomIndex) = pointLabel;
    %pointFeat(:,:,randomIndex) = pointFeat;
    pointDataNum(randomIndex) = pointDataNum;
    
    % To get size in Megabytes
    sizeOfFiles = round(sizeOfFiles/10^6);
    numberOutputFiles = ceil(sizeOfFiles/maxFileSize);
    
    blockIndexFiles = ceil(linspace(0,numberOfBlocks,numberOutputFiles+1));
    
    saveDestination = [outputFolder,fileName];
    
    fileNumbers = (1:numberOutputFiles)';
    
    
    for ii=1:numberOutputFiles
        
        saveName = [saveDestination(1:end-3),'_',num2str(fileNumbers(ii)),saveDestination(end-2:end)];
        
        delete(saveName);
        
        lowIndex = (blockIndexFiles(ii)+1);
        highIndex = blockIndexFiles(ii+1);
        
        numberOfBlocks = highIndex-blockIndexFiles(ii);
        
        h5create(saveName,'/data',[size(pointCoord,1) tileBlockPointNumber numberOfBlocks],'Datatype', dataTypesToSave(1));
        h5create(saveName,'/label',numberOfBlocks, 'Datatype',dataTypesToSave(2));
        h5create(saveName,'/label_seg',[tileBlockPointNumber numberOfBlocks], 'Datatype',dataTypesToSave(3));
        %h5create(saveName,'/normal',[3 tileBlockPointNumber numberOfBlocks],'Datatype',dataTypesToSave(4));
        
        h5write(saveName,'/data',pointCoord(:,:,lowIndex:highIndex) );
        h5write(saveName,'/label',blockLabel(lowIndex:highIndex) );
        h5write(saveName,'/label_seg',pointLabel(:,lowIndex:highIndex) );
        %h5write(saveName,'/normal',pointFeat(:,:,lowIndex:highIndex) );
        
        h5create(saveName,'/data_num',numberOfBlocks, 'Datatype','int32');
        h5write(saveName,'/data_num',pointDataNum(lowIndex:highIndex) );
        
    end
    
end

