function [intensityVisualization] = intensityPlot(intensity,resolution)
% This is a function to make a more vizualize picture of the intensity. It
% divides the intensity values to an equal destribution for each value of
% the resolution. 

    % Allocates memory for the intensity return.
    intensityVisualization = zeros(1,length(intensity));
    
    % Sort the intensity value to find an equal distrubution.
    quant = sort(intensity);

    for i= 1:(resolution-1)

        intensityVisualization( ...
        ( quant(round((i)*length(quant)/resolution)) < intensity ) & ...   
        (intensity) <= quant(floor((i+1)*length(quant)/resolution)) ) = i/resolution;
    
    end


end

