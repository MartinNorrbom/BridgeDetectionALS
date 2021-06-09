import os
import sys
import numpy as np

import accessDataFiles
import analys_functions

import filterFunctions

import training_functions
import pptk
from laspy.file import File


def preparation(filename,overlap=1,coordinateList = [],saveFolder="",savePrepFiles = 0,saveH5 = 0):
    '''This function read the data to analyse and perform overlapping with voting'''

    # Check if there is a single file or a list of files.
    if( isinstance(filename,list) ):

        # Define lists.
        list_data = []
        list_label_seg = []
        list_pred_label_seg = []

        # Loop through all files that in the list and treat it as a group.
        for i in range(len(filename)):

            # Read h5 files
            data,label_block,label_seg,pred_label,pred_label_seg,geo_coord = \
                accessDataFiles.load_h5_analys_data(filename[i])

            ##################### Overlap and voting #######################
            if( overlap ):
                data,pred_label_seg,label_seg = filterFunctions.voting_overlapping(data,pred_label_seg,geo_coord,label_seg)

            ##################### Analysis For Block Classification #######################
            # Create a text file that contains the coordinates that have been missclassificated.
            if( len(geo_coord) > 0 and coordinateList != [] ):
                saveNameCoord = saveFolder+"coordinates"+str(coordinateList)+"_"+str(i)+".txt"
                analys_functions.saveCoordinatesText(saveNameCoord,geo_coord,label_block,pred_label)
            ###############################################################################

            # Save prepared data in list
            list_data.append(data)
            list_label_seg.append(label_seg)
            list_pred_label_seg.append(pred_label_seg)

        # Merge all the prepared data from different files.
        prep_data = np.concatenate(list_data,axis=0)
        prep_label_seg = np.concatenate(list_label_seg)
        prep_pred_label_seg = np.concatenate(list_pred_label_seg)

        ##################### Overlap and voting #######################
        # Final voting if there are neighbouring files.
        if( overlap ):
            prep_data,prep_pred_label_seg,prep_label_seg = filterFunctions.voting_overlapping(prep_data,prep_pred_label_seg,[],prep_label_seg)

    else:

        # Read h5 file
        data,label_block,label_seg,pred_label,pred_label_seg,geo_coord = \
            accessDataFiles.load_h5_analys_data(filename)

        ##################### Overlap and voting #######################
        if( overlap ):
            prep_data,prep_pred_label_seg,prep_label_seg = filterFunctions.voting_overlapping(data,pred_label_seg,geo_coord,label_seg)

        ##################### Analysis For Block Classification #######################
        # Create a text file that contains the coordinates that have been missclassificated.
        if( len(geo_coord) > 0 and coordinateList != [] ):
            saveNameCoord = saveFolder+"coordinates"+str(coordinateList)+".txt"
            analys_functions.saveCoordinatesText(saveNameCoord,geo_coord,label_block,pred_label)
        ###############################################################################

    if(saveH5):
        accessDataFiles.save_results_h5(saveFolder+"Final_Voting_h5/"+"res_voting"+str(coordinateList)+".h5",prep_data,prep_label_seg,prep_pred_label_seg)
            
    # Return all point coordinates, point label, and point prediction.
    return prep_data,prep_label_seg,prep_pred_label_seg





def analys_ALS(filename,logpath, useFilter = 0,includeImage = 0, saveFolder = "", savePrep = 0, saveFiltered = 0 ):

    # Get number of grouped files to analyse
    nrFiles = len(filename)

    print("Number of grouped files to analys: " + str(nrFiles) )

    # Create a text log to save the analysis results.
    result_log = open(saveFolder+'log_results.txt', 'w')


    # Allocate empty lists.
    list_pred_seg = []
    list_label_seg = []
    list_bridge_info_C = []
    list_bridge_info_L = []
    list_bridge_info_P = []

    # Allocate analysis variables.
    countFileWithBridge = 0
    totNrBridge = 0
    totNrBridgeFound = 0


    ##################### Get Learning curve #######################
    if(len(logpath) != 0):
        analys_functions.learningCurvePlot( logpath+'log_train.txt',saveFolder+'learning_curve.png')

    ################# Loop through grouped files ###################

    for cfile in range(nrFiles):

        print("Working on file "+str(cfile))

        ###################### Voting #######################

        # Run internal voting for each tile block and voting for the overlapping.
        data,label_seg,pred_label_seg = preparation( filename[cfile], coordinateList = cfile, saveFolder=saveFolder,saveH5 = savePrep )


        ##################### Filters #######################

        if(useFilter == 1):

            # Use the selected filter parameters.
            pred_label_seg,bridgeCoord = np.copy( filterFunctions.pointFilter(data,pred_label_seg,minimumArea =10,minimumPoints = 15,searchRadius =2) )

            # Save the coordinates of the predicted bridges in text file format.
            saveNameCoord = saveFolder+"bridge_coordinates"+str(cfile)+".txt"
            analys_functions.saveCoordinatesText(saveNameCoord,np.flip(bridgeCoord,axis=1))

            # Save the filtered data in h5 format.
            if(saveFiltered):
                accessDataFiles.save_results_h5(saveFolder+"Final_Filter_h5/"+"res_filtering"+str(cfile)+".h5",data,label_seg,pred_label_seg)

        else:
            pred_label_seg = np.copy(pred_label_seg)



        ###################### Analysis per bridge ########################

        # Group the labelled bridges points into bridges.
        nrBridgesFound,nrBridges,bridgeInfo = analys_functions.CountLabelledBridges(data,label_seg,pred_label_seg)
        print("Number of bridges found: " + str(nrBridgesFound))
        print("Total number of bridges: " + str(nrBridges))

        # Count the number of predicted bridges and create a histogram.
        if( len(bridgeInfo) != 0 ):
            analys_functions.bridgeHistogram(bridgeInfo[1],bridgeInfo[2],saveFolder+'bridge_histogram_'+str(cfile)+'.png')

        # Save point cloud images of the bridges, where the colours that corresponds to the prediction.
        # (Red = TP, Yellow = FP, Green = TN, Blue = FN)
        if ( includeImage == 1 ):
            for i in range( len(bridgeInfo[0]) ):
                analys_functions.point_cloud_3D_view_plot( bridgeInfo[0][i],bridgeInfo[1][i],bridgeInfo[2][i], i )


        ###################### Analysis For Point Segmentation ########################

        label_seg_total = np.asarray(label_seg).reshape(-1)
        pred_label_seg_total = np.asarray(pred_label_seg).reshape(-1)

        # Log file name and number of bridges found.
        result_log.write( str(filename[cfile]) + "\n")
        result_log.write("Number of bridges found: " + str(nrBridgesFound) + " of " + str(nrBridges) + "\n")

        # Count bridge for all the grouped files.
        totNrBridge = nrBridges + totNrBridge
        totNrBridgeFound = nrBridgesFound + totNrBridgeFound
        
        if( len(bridgeInfo) != 0 ):

            # Get scores from the analysis methods.
            youdenScore_Seg, precision_seg, recall_seg,OA = analys_functions.analys_score_methods(label_seg_total,pred_label_seg_total)

            # Plot confusion matrix.
            analys_functions.confusion_matrix_plot(label_seg_total,pred_label_seg_total,saveFolder+"ConfusionMatrix_Segmentaion_"+str(cfile)+".png")

            # Log results, (OA, Youden, Precision, Recall)

            result_log.write('Overall accuracy: '+str(OA) + "\n")
            result_log.write('Youdens index value for points: '+str(youdenScore_Seg) + "\n")
            result_log.write('Precision value for points: '+str(precision_seg) + "\n")
            result_log.write('Recall value for points: '+str(recall_seg) + "\n")
            result_log.flush()

            # Print the results.
            print('Youdens index value for points: '+str(youdenScore_Seg))
            print('Precision value for points: '+str(precision_seg))
            print('Recall value for points: '+str(recall_seg))
        else:
            print("No statistic results for files without bridges.")

        # Append the predictions from each file groups into the lists
        if( len(bridgeInfo) != 0 ):

            
            if( len(bridgeInfo[1]) == 1 ):

                list_bridge_info_C.append(bridgeInfo[0][0])
                list_bridge_info_L.append(bridgeInfo[1][0])
                list_bridge_info_P.append(bridgeInfo[2][0])

            else:
                for i in range(len(bridgeInfo[1])):
                    list_bridge_info_C.append(bridgeInfo[0][i])
                    list_bridge_info_L.append(bridgeInfo[1][i])
                    list_bridge_info_P.append(bridgeInfo[2][i])

            countFileWithBridge = countFileWithBridge+1

        list_pred_seg.append(pred_label_seg)
        list_label_seg.append(label_seg)

        result_log.write("\n\n")

    ################## Overall results for all the files ##################

    # Get total results of the predictions.
    tot_pred = np.concatenate(list_pred_seg)
    tot_label = np.concatenate(list_label_seg)

    # Get scores from the analysis methods.
    youdenScore_Seg, precision_seg, recall_seg,totOA = analys_functions.analys_score_methods(tot_label,tot_pred)

    # Get the normalised confusion matrix.
    analys_functions.confusion_matrix_plot(tot_label,tot_pred,saveFolder+"ConfusionMatrix_Segmentaion_Total.png")

    # Print all the scores.
    result_log.write("Total test area. \n")
    result_log.write("Number of bridges found: " + str(totNrBridgeFound) + " of " + str(totNrBridge) + "\n")

    result_log.write('Overall accuracy: '+str(totOA) + "\n")
    result_log.write('Youdens index value for points: '+str(youdenScore_Seg) + "\n")
    result_log.write('Precision value for points: '+str(precision_seg) + "\n")
    result_log.write('Recall value for points: '+str(recall_seg) + "\n")


    # Get the total bridge histogram.
    if( len(list_bridge_info_L) != 0 ):

        analys_functions.bridgeHistogram(list_bridge_info_L,list_bridge_info_P,saveFolder+'bridge_histogram_total.png')
    


    ###############################################################################





def main():

    # Test files with 0% overlap.
    fileList_00 = [\
        "TestOverlap/Result_00/Result_B70_P8192_G4_TestSet_Karlstad.h5",\
        "TestOverlap/Result_00/Result_B70_P8192_G4_TestSet_Halsingborg.h5",\
        "TestOverlap/Result_00/Result_B70_P8192_G4_TestSet_Lund.h5",\
        "TestOverlap/Result_00/Result_B70_P8192_G4_TestSet_Norrkoping.h5",\
        "TestOverlap/Result_00/Result_B70_P8192_G4_TestSet_Nykoping.h5",\
        "TestOverlap/Result_00/Result_B70_P8192_G4_TestSet_Trollhattan.h5",\
        "TestOverlap/Result_00/Result_B70_P8192_G4_TestSet_Umea.h5",\
    ]

    # Test files with 25% overlap.
    fileList_25 = [\
        "TestOverlap/Result_25/Result_B70_P8192_G4_TestSet_Karlstad.h5",\
        "TestOverlap/Result_25/Result_B70_P8192_G4_TestSet_Halsingborg.h5",\
        "TestOverlap/Result_25/Result_B70_P8192_G4_TestSet_Lund.h5",\
        "TestOverlap/Result_25/Result_B70_P8192_G4_TestSet_Norrkoping.h5",\
        "TestOverlap/Result_25/Result_B70_P8192_G4_TestSet_Nykoping.h5",\
        "TestOverlap/Result_25/Result_B70_P8192_G4_TestSet_Trollhattan.h5",\
        "TestOverlap/Result_25/Result_B70_P8192_G4_TestSet_Umea.h5",\
    ]


    # Test files with 50% overlap.
    fileList_50 = [\
    [  \
        "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Halsingborg_1.h5",     \
        "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Halsingborg_2.h5"],    \
        "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Karlstad.h5",          \
    [   "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Lund_1.h5",            \
        "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Lund_2.h5",            \
        "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Lund_3.h5",            \
        "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Lund_4.h5"],           \
    [   "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Norrkoping_1.h5",      \
        "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Norrkoping_2.h5",      \
        "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Norrkoping_3.h5",      \
        "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Norrkoping_4.h5"],     \
    [   "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Nykoping_1.h5",        \
        "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Nykoping_2.h5",        \
        "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Nykoping_3.h5",        \
        "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Nykoping_4.h5"],       \
    [   "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Trollhattan_1.h5",     \
        "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Trollhattan_2.h5",     \
        "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Trollhattan_3.h5",     \
        "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Trollhattan_4.h5"],    \
        "TestOverlap/Result_50/Result_B70_P8192_G4_TestSet_Umea.h5"]

    # Test files with 50% overlap, after voting.
    fileList_50_Voted = [\
        "Results_Analysis/Overlap_50/Final_Voting_h5/res_voting0.h5",    \
        "Results_Analysis/Overlap_50/Final_Voting_h5/res_voting1.h5",    \
        "Results_Analysis/Overlap_50/Final_Voting_h5/res_voting2.h5",    \
        "Results_Analysis/Overlap_50/Final_Voting_h5/res_voting3.h5",    \
        "Results_Analysis/Overlap_50/Final_Voting_h5/res_voting4.h5",    \
        "Results_Analysis/Overlap_50/Final_Voting_h5/res_voting5.h5",    \
        "Results_Analysis/Overlap_50/Final_Voting_h5/res_voting6.h5"]

    # Folder for the files saved after the analysis.
    sFolders = ["Results_Analysis/Overlap_00/","Results_Analysis/Overlap_25/","Results_Analysis/Overlap_50/"]

    # Analyse 0, 25, and 50 % overlap.
    analys_ALS(fileList_00,[],useFilter=0,includeImage=0,saveFolder=sFolders[0])

    analys_ALS(fileList_25,[],useFilter=0,includeImage=0,saveFolder=sFolders[1])

    analys_ALS(fileList_50_Voted,[],useFilter=1,includeImage=0,saveFolder=sFolders[2],saveFiltered=0)




if __name__ == "__main__":
    main()

