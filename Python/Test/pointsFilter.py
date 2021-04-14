
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


def pointFilter(coordinates,pred_label_seg):

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
        clustering = DBSCAN(eps=2, min_samples=1).fit(clusterCoord)

        # Loop through all the clusters.
        for i in range(max(clustering.labels_)+1):

            # Get the XY coordinates from the points in the current cluster.
            pointsHull = np.copy(clustering.components_[clustering.labels_ == i, 0:2])

            # Get unique XY coordinates
            uniqueValues,uniqueIndex = np.unique(pointsHull,axis=0,return_index=True)

            # At least 3 unique points is needed to get an area over the cluster.
            if( len(uniqueIndex) >= 3 ):

                # Use convex hull to find the area of XY coordinates over the cluster.
                hull = ConvexHull( pointsHull,'volume')

                # # Plot the cluster Debug
                # plt.plot(clustering.components_[:,0], clustering.components_[:,1], 'o')

                # for simplex in hull.simplices:

                #     plt.plot(pointsHull[simplex, 0], pointsHull[simplex, 1], 'k-')
                # print("Area is: "+str(hull.volume))
                # plt.show()

                if( hull.volume < 3 ):
                    filtered_predication[ index[clustering.labels_ == i] ] = 0

                    print("Area to small.")
                    print("Deleted cluster: "+str(i))

            else:

                filtered_predication[ index[clustering.labels_ == i] ] = 0
                print("To few points. Number of points: "+str(len(pointsHull)))
                print("Deleted cluster: "+str(i))

    
    
    return filtered_predication





def main():
    
    data,label_block,label_seg,pred_label,pred_label_seg,geo_coord = \
    accessDataFiles.load_h5_analys_data("Result/Result_B30_P1024_G4_debugTest_Mfiles.h5")

    block = 31

    pointFilter(data[block,:,0:3],pred_label_seg[block,:])





if __name__ == "__main__":
    main()
