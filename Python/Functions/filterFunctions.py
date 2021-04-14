
import os
import scipy.misc
import sys
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, convex_hull_plot_2d

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

sys.path.insert(1, '../Functions')

import accessDataFiles
import analysis_ALS


def pointFilter(coordinates,pred_label_seg, minimumArea = 3, minimumPoints = 3, searchRadius = 2):

    # Indicies classified as bridges.
    positiveIndex = pred_label_seg==1

    # Change the invalid bridge points, predicted from the ML algorithm.
    filtered_predication = np.copy(pred_label_seg)

    if(sum(positiveIndex) > 0):

        # Get numeric indecies.
        index = np.zeros(np.sum(positiveIndex)).astype(int)
        for i in range(len(index)):
            if(positiveIndex[i] == 1):
                index[i] = i
        
        # Get XYZ coordinates for only bridge points.
        clusterCoord = np.copy( coordinates[positiveIndex,:] )

        
        # Use DBSCAN to get clusters of bridge points.
        clustering = DBSCAN(eps=searchRadius, min_samples=1).fit(clusterCoord)

        # Loop through all the clusters.
        for i in range(max(clustering.labels_)+1):

            # Get the XY coordinates from the points in the current cluster.
            pointsHull = np.copy(clustering.components_[clustering.labels_ == i, 0:2]).astype(dtype='float64')

            # Get unique XY coordinates
            uniqueValuesX,uniqueIndexX = np.unique(pointsHull[:,0],axis=0,return_index=True)
            uniqueValuesY,uniqueIndexY = np.unique(pointsHull[:,1],axis=0,return_index=True)

            # At least 3 unique points is needed to get an area over the cluster.
            if( len(uniqueIndexX) >= minimumPoints and len(uniqueIndexY) >= minimumPoints ):

                # Use convex hull to find the area of XY coordinates over the cluster.
                hull = ConvexHull( pointsHull,'volume','QJ')


                if( hull.volume < minimumArea ):
                    filtered_predication[ index[clustering.labels_ == i] ] = 0

            else:

                filtered_predication[ index[clustering.labels_ == i] ] = 0
    
    
    return filtered_predication





def main():


    data,label_block,label_seg,pred_label,pred_label_seg,geo_coord = \
    accessDataFiles.load_h5_analys_data("Result/Result_B30_P1024_G4_debugTest_Mfiles.h5")

    block = 31

    pointFilter(data[block,:,0:3],pred_label_seg[block,:])


if __name__ == "__main__":
    main()
