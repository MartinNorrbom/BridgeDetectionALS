function [sizeOfFiles] = mixH5Files(fileName,inputFolder,outputFolder,maxFileSize)
%mixH5Files Summary of this function goes here
%   Detailed explanation goes here

    selectFilter = [inputFolder,'*.h5'];

    [file,path] = uigetfile( selectFilter,'MultiSelect','on');
    
    sizeOfFiles = 0;
    
    if(iscell(file))
        nrOfFiles = length(file);
        
        for ii=1:nrOfFiles
            selectedFiles = dir([path,file{ii}]);
            sizeOfFiles = sizeOfFiles + selectedFiles.bytes;
        end
        
    else
        nrOfFiles = 1;
        selectedFiles = dir([path,file]);
        sizeOfFiles =  selectedFiles.bytes;
        
        file = {file};
    end
    
    

    
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

