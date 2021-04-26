import os
import sys
import numpy as np

import accessDataFiles
import analys_functions

import filterFunctions

import training_functions
import pptk



def preparation(filename,overlap=1,coordinateList = [],saveFolder="",savePrepFiles = 0):
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
        prep_data = np.concatenate(list_data)
        prep_label_seg = np.concatenate(list_label_seg)
        prep_pred_label_seg = np.concatenate(list_pred_label_seg)

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
            
    # Return all point coordinates, point label, and point prediction.
    return prep_data,prep_label_seg,prep_pred_label_seg





def analys_ALS(filename,logpath, useFilter = 0,includeImage = 0, saveFolder = ""):

    nrFiles = len(filename)

    print("Numbers of files to analys: " + str(nrFiles) )

    result_log = open(saveFolder+'log_results.txt', 'w')


    list_pred_seg = []
    list_label_seg = []

    list_bridge_info_C = []
    list_bridge_info_L = []
    list_bridge_info_P = []

    countFileWithBridge = 0

    ##################### Get Learning curve #######################
    if(len(logpath) != 0):
        analys_functions.learningCurvePlot( logpath+'log_train.txt',saveFolder+'learning_curve.png')


    for cfile in range(nrFiles):


        data,label_seg,pred_label_seg = preparation( filename[cfile], coordinateList = cfile, saveFolder=saveFolder )


        ##################### Filters #######################

        if(useFilter == 1):
            pred_label_seg = filterFunctions.pointFilter(data,pred_label_seg,10,20,3)


        ###################### Analysis per bridge ########################

        nrBridgesFound,nrBridges,bridgeInfo = analys_functions.CountLabelledBridges(data,label_seg,pred_label_seg)
        print("Number of bridges found: " + str(nrBridgesFound))
        print("Total number of bridges: " + str(nrBridges))

        if( len(bridgeInfo) != 0 ):
            analys_functions.bridgeHistogram(bridgeInfo[1],bridgeInfo[2],saveFolder+'bridge_histogram_'+str(cfile)+'.png')

        if ( includeImage == 1 ):
            for i in range( len(bridgeInfo[0]) ):
                analys_functions.point_cloud_3D_view_plot( bridgeInfo[0][i],bridgeInfo[1][i],bridgeInfo[2][i], i )


        ###################### Analysis For Point Segmentation ########################

        label_seg_total = np.asarray(label_seg).reshape(-1)
        pred_label_seg_total = np.asarray(pred_label_seg).reshape(-1)

        # Log file name.
        result_log.write( str(filename[cfile]) + "\n")
        result_log.write("Number of bridges found: " + str(nrBridgesFound) + " of " + str(nrBridges) + "\n")

        
        if( len(bridgeInfo) != 0 ):

            # Get scores.
            youdenScore_Seg, precision_seg, recall_seg = analys_functions.analys_score_methods(label_seg_total,pred_label_seg_total)

            # Plot confusion matrix.
            analys_functions.confusion_matrix_plot(label_seg_total,pred_label_seg_total,saveFolder+"ConfusionMatrix_Segmentaion_"+str(cfile)+".png")

            # Log results
            result_log.write('Youdens index value for points: '+str(youdenScore_Seg) + "\n")
            result_log.write('Precision value for points: '+str(precision_seg) + "\n")
            result_log.write('Recall value for points: '+str(recall_seg) + "\n")

            # Print Youden's J statistics and Cohen's kappa.
            print('Youdens index value for points: '+str(youdenScore_Seg))
            print('Precision value for points: '+str(precision_seg))
            print('Recall value for points: '+str(recall_seg))
        else:
            print("No statistic results for files without bridges.")


        # Create images
        # if (includeImage):
        #     for i in range(len(label_block)):
        #         if label_block[i] != pred_label[i]:

        #             analys_functions.point_cloud_3D_view_plot( data[i,:,:],label_seg[i,:],pred_label_seg[i,:], i )


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

    tot_pred = np.concatenate(list_pred_seg)
    tot_label = np.concatenate(list_label_seg)


    youdenScore_Seg, precision_seg, recall_seg = analys_functions.analys_score_methods(tot_label,tot_pred)

    analys_functions.confusion_matrix_plot(tot_label,tot_pred,saveFolder+"ConfusionMatrix_Segmentaion_Total.png")

    result_log.write( "Total test area. \n")
    result_log.write("Number of bridges found: " + str(nrBridgesFound) + " of " + str(nrBridges) + "\n")

    result_log.write('Youdens index value for points: '+str(youdenScore_Seg) + "\n")
    result_log.write('Precision value for points: '+str(precision_seg) + "\n")
    result_log.write('Recall value for points: '+str(recall_seg) + "\n")



    if( len(list_bridge_info_L) != 0 ):


        analys_functions.bridgeHistogram(list_bridge_info_L,list_bridge_info_P,saveFolder+'bridge_histogram_total.png')
    




    ###############################################################################





def main():
    #analys_ALS(["../Test/TestResults/result_step_3_B70_P8192_6m12m/Karlstad/Result_B70_P8192_Karlstad_Test_Set.h5"],"../TrainedModels/step_4_B70_P8192_check/")

    #analys_ALS(["../Test/TestResults/result_step_3_B70_P8192_6m12m/OnlyBridge/Result_B70_P8192_OnlyBridge_Test_Set.h5"],"../TrainedModels/step_4_B70_P8192_check/")

    #analys_ALS(["../Test/TestResults/result_step_3_B70_P8192_6m12m/OnlyBridge/Result_B70_P8192_OnlyBridge_Test_Set.h5"],"../TrainedModels/")

    # analys_ALS(["Results\Result_00\Result_B70_P8192_G4_TestSet_Karlstad.h5"],"..\\TrainedModels\\20210401_B30_P1024_5F\\",useFilter=1,includeImage=0)

    fileList_00 = [\
        "TestOverlap/Result_00/Result_B70_P8192_G4_TestSet_Karlstad.h5",\
        "TestOverlap/Result_00/Result_B70_P8192_G4_TestSet_Halsingborg.h5",\
        "TestOverlap/Result_00/Result_B70_P8192_G4_TestSet_Lund.h5",\
        "TestOverlap/Result_00/Result_B70_P8192_G4_TestSet_Norrkoping.h5",\
        "TestOverlap/Result_00/Result_B70_P8192_G4_TestSet_Nykoping.h5",\
        "TestOverlap/Result_00/Result_B70_P8192_G4_TestSet_Trollhattan.h5",\
        "TestOverlap/Result_00/Result_B70_P8192_G4_TestSet_Umea.h5",\
    ]


    fileList_25 = [\
        "TestOverlap/Result_25/Result_B70_P8192_G4_TestSet_Karlstad.h5",\
        "TestOverlap/Result_25/Result_B70_P8192_G4_TestSet_Halsingborg.h5",\
        "TestOverlap/Result_25/Result_B70_P8192_G4_TestSet_Lund.h5",\
        "TestOverlap/Result_25/Result_B70_P8192_G4_TestSet_Norrkoping.h5",\
        "TestOverlap/Result_25/Result_B70_P8192_G4_TestSet_Nykoping.h5",\
        "TestOverlap/Result_25/Result_B70_P8192_G4_TestSet_Trollhattan.h5",\
        "TestOverlap/Result_25/Result_B70_P8192_G4_TestSet_Umea.h5",\
    ]


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


    sFolders = ["Results_Analysis/Overlap_00/","Results_Analysis/Overlap_25/","Results_Analysis/Overlap_50/"]


#    analys_ALS(fileList_00,"../TrainedModels/step_4_B70_P8192_check/",useFilter=1,includeImage=0,saveFolder=sFolders[0])

#    analys_ALS(fileList_25,"../TrainedModels/step_4_B70_P8192_check/",useFilter=1,includeImage=0,saveFolder=sFolders[1])

    analys_ALS(fileList_50,[],useFilter=0,includeImage=0,saveFolder=sFolders[2])



if __name__ == "__main__":
    main()

