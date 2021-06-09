
import os
import scipy.misc
import sys
import numpy as np
#import gnumpy as npg
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, convex_hull_plot_2d

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

sys.path.insert(1, '../Functions')

import accessDataFiles
import analysis_ALS
import time
import pptk


def pointFilter(coordinates,pred_label_seg, minimumArea = 3, minimumPoints = 3, searchRadius = 2, pointDensity = 0):

    # Indicies classified as bridges.
    positiveIndex = pred_label_seg==1

    # Change the invalid bridge points, predicted from the ML algorithm.
    filtered_predication = np.copy(pred_label_seg)

    if(sum(positiveIndex) > 0):

        indCount = 0

        # Get numeric indecies.
        index = np.zeros(np.sum(positiveIndex)).astype(int)

        #index = np.where( pred_label_seg==1 ).astype(np.int64)
        for i in range(len(pred_label_seg)):
            if(pred_label_seg[i] == 1):

                index[indCount] = i
                indCount = np.copy(indCount)+1

        # Get XYZ coordinates for only bridge points.
        clusterCoord = np.copy( coordinates[positiveIndex,:] )

        
        # Use DBSCAN to get clusters of bridge points.
        clustering = DBSCAN(eps=searchRadius, min_samples=1).fit(clusterCoord)

        nrCluster = max(clustering.labels_)

        centerSavedCluster = []

        # Loop through all the clusters.
        for i in range(max(clustering.labels_)+1):

            currentClusterIndex = clustering.labels_ == i

            # Get the XY coordinates from the points in the current cluster.
            pointsHull = np.copy(clustering.components_[currentClusterIndex, 0:2]).astype(dtype='float64')

            # Get unique XY coordinates
            uniqueCoords,uindex = np.unique(pointsHull[:,0:2],axis=0,return_index=True)

            # At least 3 unique points is needed to get an area over the cluster.
            if( uniqueCoords.shape[0] >= minimumPoints ):

                # Use convex hull to find the area of XY coordinates over the cluster.
                hull = ConvexHull( pointsHull,'volume','QJ')

                # Remove clusters less than the required covered area.
                if( hull.volume < minimumArea ):
                    filtered_predication[ index[currentClusterIndex] ] = 0


                # Remove clusters with to low density.
                elif( (hull.volume/uniqueCoords.shape[0]) < pointDensity ):
                    filtered_predication[ index[currentClusterIndex] ] = 0

                else:

                    centerSavedCluster.append( np.squeeze(np.round( np.average(uniqueCoords,axis=0) ).astype(np.int)) )

            else:
                filtered_predication[ index[currentClusterIndex] ] = 0

    




    return filtered_predication,centerSavedCluster


def voting_overlapping(coordinates,pred_label_seg, geo_coord, label_seg = [] ):



    # Create lists
    pointList = []
    predList = []
    pointGeoCoordTB = []

    labelListTB = []

    startVT = time.time()

    # 1. Make intern voting for each interpolated tile block.
    # 2. Add the center coordinate to all the blocks to get orginal coordinates.

    # Check if there is multiple tile blocks. If so do step 1.
    if( len(coordinates.shape) == 3 ):

        # Get number of tile blocks and number of points.
        nrBlocks = coordinates.shape[0]
        nrPoints = coordinates.shape[1]

        # Loop through all tile blocks.
        for i in range(nrBlocks):

            # Get unique coordinates and indecies.
            tempCoord,index_to_use,index,uCounts = np.unique(coordinates[i,:,:],return_index=1,return_inverse=1,return_counts=1,axis=0)

            # If the tile block has been interpolated, make intern voting.
            if( len(index_to_use) < nrPoints ):

                # Get number of unique points.
                nrRows = np.max(index) + 1

                # Allocate memory to stack all the predictions for each point.
                voteCount = np.zeros([nrRows, np.max(uCounts)],dtype=np.int8)

                # Allocate memory to count the total number of votes for each point.
                nrVoteCount = np.zeros(nrRows,dtype=np.int8)

                # Loop through all the points in the tile block.
                for j in range(len(index)):
                    # Add vote for the current unique point.
                    voteCount[ index[j], nrVoteCount[index[j]] ] = pred_label_seg[i,j]

                    # Add number of votes for the current unique point.
                    nrVoteCount[ index[j] ] = np.copy( nrVoteCount[ index[j] ] + 1 )

                # Get the voting results for each unique point.
                voteRes = np.divide(np.sum(voteCount,axis=1),nrVoteCount)

                # Allocate memory for prediction label of each point in the tile block.
                tempIndex = np.zeros(tempCoord.shape[0],dtype='uint8')

                # Set all point labels with bridge label majority to bridge.
                tempIndex[ np.where( voteRes > 0.5 ) ] = 1

                predList.append(tempIndex)
            else:
                # All points were unique so no voting is needed.
                predList.append( pred_label_seg[i,index] )

            # Add coordinates for the geographic location for all the points.
            addCoord = [np.array([ geo_coord[i,1], geo_coord[i,0], geo_coord[i,2] ]),]*len(tempCoord)

            # Add coordinates to the list.
            pointList.append( tempCoord + addCoord )

            # Add center location of current tile block.
            pointGeoCoordTB.append( addCoord )

            if(len(label_seg) != 0):
                labelListTB.append( label_seg[i,index_to_use] )

        # Merge all the stored data.
        allCoord = np.concatenate(pointList)
        allPred = np.concatenate(predList)

        allPointGeoCoord = np.concatenate(pointGeoCoordTB)

        if(len(label_seg) != 0):
            allLabel_seg = np.concatenate(labelListTB)

    else:
        # There coordinates is not divided into tile block, so all the data is added.
        allCoord = np.copy(coordinates)
        allPred = np.copy(pred_label_seg)
        allLabel_seg = np.copy(label_seg)


    endVT = time.time()
    print("Time voting tile block: ")
    print(endVT - startVT)

    startU = time.time()

    # Get all unique coordinates.
    uAllCoord,index_to_use,uAllIndex,uCounts = np.unique(allCoord,return_index=1,return_inverse=1,return_counts=1,axis=0)

    uAllIndex.astype(int)

    endU = time.time()
    print("Time unique: ")

    print("Total number of coordinates.")
    print(allCoord.shape[0])
    print("Number of unique coordinates.")
    print(uAllCoord.shape[0])

    print(endU - startU)

    startVO = time.time()

    # Check if there are any dublicates of coordinates.
    if( uAllCoord.shape[0] < allCoord.shape[0] ):

        # Get number of unique points.
        nrRows = np.max(uAllIndex)+1

        # Allocate memory to stack all the predictions for each point.
        voteCount = np.zeros([nrRows, np.max(uCounts)],dtype=np.int8)

        # Allocate memory to count the total number of votes for each point.
        nrVoteCount = np.zeros(nrRows,dtype=np.int8)

        # Index of votes
        indexVoteCount = -1*np.ones([nrRows, np.max(uCounts)],dtype=np.int64)

        # Loop through all points.
        for i in range(len(uAllIndex)):

            # Add vote for the current unique point.
            voteCount[ uAllIndex[i], nrVoteCount[uAllIndex[i]] ] = allPred[i]

            # Save vote index in case for equal voting score
            indexVoteCount[ uAllIndex[i], nrVoteCount[uAllIndex[i]] ] = i

            # Add number of votes for the current unique point.
            nrVoteCount[ uAllIndex[i] ] = np.copy( nrVoteCount[ uAllIndex[i] ] + 1 )

        # Get vote results.
        voteRes = np.divide(np.sum(voteCount,axis=1),nrVoteCount)

        # Allocate memory for predication label.
        uPred = np.zeros( uAllCoord.shape[0],dtype='uint8' )

        # If majority votes is bridges set prediction label to bridge.
        uPred[ np.where( voteRes > 0.5 ) ] = 1

        # Indicies for equal votes.
        indexEqualVotes = np.squeeze( np.where( voteRes == 0.5 ), axis=0 )

        print("Number of equal votes: ")
        print(indexEqualVotes.shape)

        # Check if there is multiple tile blocks.
        if( len(coordinates.shape) == 3 ):

            # If there is multiple blocks and there is draw voting. 
            # The majority vote will be the prediction from the tile 
            # block that is closest to the point.

            # Loop through all equal vote index.
            for i in indexEqualVotes:

                # Get index of voting points.
                tempIndex = np.copy(indexVoteCount[i, indexVoteCount[i,:] > -1 ] )

                # Get geographical coordinates of the voting tile blocks.
                tempGeoCoord = np.copy( allPointGeoCoord[tempIndex,:] )

                # Get distance between point and center of tile blocks.
                tempdist = np.sum( np.power( tempGeoCoord - uAllCoord[i,:],2 ) ,axis=1)

                # Set the prediction from closest tile block.
                uPred[i] = np.copy( voteCount[i, np.argmin(tempdist)] )


    else:
        # If all points are unique.
        uPred = np.copy(allPred[index_to_use])

    endVO = time.time()
    print("Time voting overlap: ")
    print(endVO - startVO)


    if(len(label_seg) != 0):
        uLabel_seg = np.copy( allLabel_seg[index_to_use] )
    else:
        uLabel_seg = []

    return uAllCoord,uPred,uLabel_seg

    


def main():


    data,label_block,label_seg,pred_label,pred_label_seg,geo_coord = \
    accessDataFiles.load_h5_analys_data("TestOverlap/Result_B70_P8192_G4_Karlstad_TestSet_6.h5")

    # block = 31

    # pointFilter(data[block,:,0:3],pred_label_seg[block,:])

    voting_overlapping(data,pred_label_seg,geo_coord)


if __name__ == "__main__":
    main()
