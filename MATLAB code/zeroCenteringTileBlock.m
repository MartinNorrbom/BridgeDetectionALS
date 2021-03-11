function [zeroCenteredTB] = zeroCenteringTileBlock(ptCloudTB,centerCoord)
%  This function is used to zero center the point cloud tile block   
%   Detailed explanation goes here:
%   Input: 
%       ptCloudTB: Input tile block point cloud
%       centerCoord: The coordinates of the the center point for tile block
%   Output:
%       zeroCenteredTB: zero Centered XYZ coordinates of tile block 

    zeroCenteredTB = ptCloudTB;
    
    % zero centering for x,y coordinates
    zeroCenteredTB(1,:) = zeroCenteredTB(1,:) - centerCoord(1);
    zeroCenteredTB(2,:) = zeroCenteredTB(2,:) - centerCoord(2);
    
    if(length(centerCoord) < 3)
        % If there is no z coordinate in the input centerCoord
        zeroCenteredTB(3,:) = zeroCenteredTB(3,:) - median(zeroCenteredTB(3,:));
        
    else
        % If there is z coordinate in the input centerCoord
        zeroCenteredTB(3,:) = zeroCenteredTB(3,:) - centerCoord(3);
        
    end
    
end

