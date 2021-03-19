import numpy as np
#import pptk
import h5py 
import importlib
import math



def saveCoordinatesText(fileName,geo_coord,label_block,pred_label):
    ''' This function saves the coordinate of the missclassified tile blocks in a text file. '''

    # Get indecies of missclassified tile blocks.
    indexCoord = label_block != pred_label

    # Get the coordinates over the missclassified tile blocks.
    coordsToSave = geo_coord[indexCoord,:]

    # Write the coordinates in the file.
    np.savetxt(fileName,coordsToSave,fmt='%0.02d')

    print(str(coordsToSave.shape))
