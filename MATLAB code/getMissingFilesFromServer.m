function getMissingFilesFromServer(filePathsAndNames,path1Server,varargin)
%getMissingFilesFromServer Summary of this function goes here
%   Detailed explanation goes here

    nrInputs = 2;
    extraInputs = nargin - nrInputs;

    if(extraInputs>nrInputs)
        dataLAZPath = varargin{1};
    else
        dataLAZPath = [];
    end


    % Get all the file names for the LAZ files located in the data folder.
    dataFolderLAZ = dir(fullfile(dataLAZPath,'*.laz'));
    allFileNames = cell2mat({dataFolderLAZ.name});

    if(isempty(allFileNames))
        allFileNames = '';
    end

    noCatch = true;
    cFolder = path1Server;
    ftpobj = [];

    % Loop through all the regions.
    for ii=1:size(filePathsAndNames{1},1)

        % Check if there are index boxes located in the current region.
        if( ~isempty( filePathsAndNames{1}{ii} ) )

            % If at least one file is missing and the current index box is 
            % missing, locate the path to the current index box in the server.
            if (noCatch == false)
                if ~contains(cFolder,[path1Server,filePathsAndNames{1}{ii}] )
                    cFolder = [path1Server,filePathsAndNames{1}{ii}];
                    cd(ftpobj,cFolder);
                end
            end

            lengthJJ = size(filePathsAndNames{2}{ii},1);
            for jj=1:lengthJJ

                % Download the LAZ file if it is not downloaded.
                if( ~contains(allFileNames,filePathsAndNames{2}{ii}(jj,:)) )

                    % Check if an connection to the ftp server is establish.
                    if(noCatch)
                        noCatch = false;
                        disp("At least one file is missing. Log in to download the files.")

                        % Write user name and password for the ftp-server.
                        UserName = input('Write user name: \n','s');
                        Password = input('Write password: \n','s');

                        % Connect and login to the ftp server
                        ftpobj = ftp(pathWeb,UserName,Password);

                        % Get to the folder in the ftp server where the missing 
                        % index box is located.
                        cFolder = [path1Server,filePathsAndNames{1}{ii}];
                        cd(ftpobj,cFolder);
                    end

                    % Download missing index box
                    disp(['Download: ',filePathsAndNames{2}{ii}(jj,:)])
                    mget(ftpobj,filePathsAndNames{2}{ii}(jj,:),dataLAZPath);

                end
            end
        end
    end

    % Close ftp server connection.
    if (noCatch == false)
        close(ftpobj);
    end

end

