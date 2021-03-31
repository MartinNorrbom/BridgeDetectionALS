function getMissingFilesFromServer(filePathsAndNames,serverName,path1Server,varargin)
%getMissingFilesFromServer downloads LAZ-files from "Lantmäteriet's" server
% that missing.

%   Input: 
%       filePathsAndNames: Should be a cell array that contains all the
%       paths to find the LAZ-files in the server and the names of the
%       files. The size is "number of paths"x2, where the rows in the first
%       column contains the paths. Each cell in the second column should
%       contain all the LAZ-files needed that are located path of the same
%       row in column 1.
%       serverName: Is the web-address to the server.
%       path1Server: Is the path to the folder where all the LAZ-data is stored.
%   Extra Inputs: The path for the destination where all the LAZ-file will 
%       be downloaded and checked if the files already exist.

    % The number of required inputs.
    nrInputs = 3;
    % The number of extra inputs.
    extraInputs = nargin - nrInputs;

    % If there are any extra inputs it will be used as path to folder for
    % download.
    if(extraInputs>0)
        dataLAZPath = varargin{1};
    else
        dataLAZPath = [];
    end


    % Get all the file names for the LAZ files located in the data folder.
    dataFolderLAZ = dir(fullfile(dataLAZPath,'*.laz'));
    allFileNames = cell2mat({dataFolderLAZ.name});

    % If there are no LAZ-files.
    if(isempty(allFileNames))
        allFileNames = '';
    end

    % Trigger for login to server.
    noCatch = true;
    % Variable to store path in server.
    cFolder = path1Server;
    % Variable to handle FTP servers.
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
                    contents = dir(ftpobj);
                end
            end
            % Get the number of files that needs to be downloaded in
            % current path.
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
                        disp([repmat(char(8), 1, length(Password)+1)])

                        % Connect and login to the ftp server
                        ftpobj = ftp(serverName,UserName,Password);

                        % Get to the folder in the ftp server where the missing 
                        % index box is located.
                        cFolder = [path1Server,filePathsAndNames{1}{ii}];
                        cd(ftpobj,cFolder);
                        contents = dir(ftpobj);
                    end

                    % Download missing index box
                    disp(['Download: ',filePathsAndNames{2}{ii}(jj,:)])
                    
                    % Check if file is available in server.
                    if( contains( strcat(contents(:).name),filePathsAndNames{2}{ii}(jj,:) ) )
                        mget(ftpobj,filePathsAndNames{2}{ii}(jj,:),dataLAZPath);
                    else
                        disp(['File not found in server: ',filePathsAndNames{2}{ii}(jj,:)])
                    end

                end
            end
        end
    end

    % Close ftp server connection.
    if (noCatch == false)
        close(ftpobj);
    end

end

