import os
import sys
import numpy as np

import accessDataFiles
import returnGeoCoord


def analys_ALS(filename):

    print(len(filename))
    print(filename)
    print(filename[0])


    data,label_block,label_seg,pred_label,pred_label_seg,geo_coord = \
    accessDataFiles.load_h5_analys_data(filename[0])
    
    if( len(geo_coord) > 0 ):
        returnGeoCoord.saveCoordinatesText("coordinates.txt",geo_coord,label_block,pred_label)



