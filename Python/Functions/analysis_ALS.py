import os
import sys
import numpy as np

import accessDataFiles
import analys_functions

import filterFunctions

import training_functions
import pptk

def analys_ALS(filename,logpath, useFilter = 0,includeImage = 0):

    nrFiles = len(filename)

    print("Numbers of files to analys: " + str(nrFiles) )

    result_log = open('log_results.txt', 'w')


    list_pred_seg = []
    list_label_seg = []

    list_bridge_info = []

    for cfile in range(nrFiles):

        data,label_block,label_seg,pred_label,pred_label_seg,geo_coord = \
            accessDataFiles.load_h5_analys_data(filename[cfile])

        # pred_label = np.copy(label_block)
        # pred_label_seg = np.copy(label_seg)

        ##################### Get Learning curve #######################
        analys_functions.learningCurvePlot( logpath+'log_train.txt','learning_curve.png')


        ##################### Analysis For Block Classification #######################

        # Create a text file that contains the coordinates that have been missclassificated.
        if( len(geo_coord) > 0 ):
            analys_functions.saveCoordinatesText("coordinates"+str(cfile)+".txt",geo_coord,label_block,pred_label)

        ##################### Filters #######################

        if(useFilter == 1):

            f_data,f_pred_label_seg,f_label_seg = filterFunctions.voting_overlapping(data,pred_label_seg,geo_coord,label_seg)

            data = np.copy(f_data)

            label_seg = np.copy(f_label_seg)

            #pred_label_seg = np.copy(f_pred_label_seg)

            pred_label_seg = filterFunctions.pointFilter(f_data,f_pred_label_seg,10,20,3)

            # for i in range(data.shape[0]):

            #     pred_label_seg[i,:] = filterFunctions.pointFilter(data[i,:,0:3],pred_label_seg[i,:],10,20,3)
                
            #     tempVar = np.sum(pred_label_seg[i,:])
            #     if(tempVar == 0):
            #         pred_label[i] = 0

        ###################### Analysis per bridge ########################

        nrBridgesFound,nrBridges,bridgeInfo = analys_functions.CountLabelledBridges(data,label_seg,pred_label_seg,geo_coord)
        print("Number of bridges found: " + str(nrBridgesFound))
        print("Total number of bridges: " + str(nrBridges))

        if( len(bridgeInfo) != 0 ):
            analys_functions.bridgeHistogram(bridgeInfo[1],bridgeInfo[2],'bridge_histogram_'+str(cfile)+'.png')

        if ( includeImage == 1 ):
            for i in range( len(bridgeInfo[0]) ):
                analys_functions.point_cloud_3D_view_plot( bridgeInfo[0][i],bridgeInfo[1][i],bridgeInfo[2][i], i )


        ###################### Analysis For Point Segmentation ########################

        label_seg_total = np.asarray(label_seg).reshape(-1)
        pred_label_seg_total = np.asarray(pred_label_seg).reshape(-1)

        youdenScore_Seg, precision_seg, recall_seg = analys_functions.analys_score_methods(label_seg_total,pred_label_seg_total)

        analys_functions.confusion_matrix_plot(label_seg_total,pred_label_seg_total,"ConfusionMatrix_Segmentaion_"+str(cfile)+".png")

        
        result_log.write( str(filename[cfile]) + "\n")
        result_log.write("Number of bridges found: " + str(nrBridgesFound) + " of " + str(nrBridges) + "\n")

        result_log.write('Youdens index value for points: '+str(youdenScore_Seg) + "\n")
        result_log.write('Precision value for points: '+str(precision_seg) + "\n")
        result_log.write('Recall value for points: '+str(recall_seg) + "\n")


        # Print Youden's J statistics and Cohen's kappa.
        print('Youdens index value for points: '+str(youdenScore_Seg))
        print('Precision value for points: '+str(precision_seg))
        print('Recall value for points: '+str(recall_seg))


        # Create images
        # if (includeImage):
        #     for i in range(len(label_block)):
        #         if label_block[i] != pred_label[i]:

        #             analys_functions.point_cloud_3D_view_plot( data[i,:,:],label_seg[i,:],pred_label_seg[i,:], i )

        list_bridge_info.append(bridgeInfo)
        list_pred_seg.append(pred_label_seg)
        list_label_seg.append(label_seg)

    tot_pred = np.concatenate(list_pred_seg)
    tot_label = np.concatenate(list_label_seg)


    youdenScore_Seg, precision_seg, recall_seg = analys_functions.analys_score_methods(tot_label,tot_pred)

    analys_functions.confusion_matrix_plot(label_seg_total,pred_label_seg_total,"ConfusionMatrix_Segmentaion_Total.png")

    result_log.write( "Total test area. \n")
    result_log.write("Number of bridges found: " + str(nrBridgesFound) + " of " + str(nrBridges) + "\n")

    result_log.write('Youdens index value for points: '+str(youdenScore_Seg) + "\n")
    result_log.write('Precision value for points: '+str(precision_seg) + "\n")
    result_log.write('Recall value for points: '+str(recall_seg) + "\n")
    

    
    tot_bridge_info = np.concatenate(list_bridge_info)

    if( len(tot_bridge_info) != 0):
        analys_functions.bridgeHistogram(tot_bridge_info[1],tot_bridge_info[2],'bridge_histogram_total.png')

    analys_functions.confusion_matrix_plot(tot_label,tot_pred,"ConfusionMatrix_Segmentaion_total.png")





    ###############################################################################





def main():
    #analys_ALS(["../Test/TestResults/result_step_3_B70_P8192_6m12m/Karlstad/Result_B70_P8192_Karlstad_Test_Set.h5"],"../TrainedModels/step_3_B70_P8192_6m12m_600/")

    #analys_ALS(["../Test/TestResults/result_step_3_B70_P8192_6m12m/OnlyBridge/Result_B70_P8192_OnlyBridge_Test_Set.h5"],"../TrainedModels/step_3_B70_P8192_6m12m_600/")

    #analys_ALS(["../Test/TestResults/result_step_3_B70_P8192_6m12m/OnlyBridge/Result_B70_P8192_OnlyBridge_Test_Set.h5"],"../TrainedModels/")

    # analys_ALS(["Results\Result_00\Result_B70_P8192_G4_TestSet_Karlstad.h5"],"..\\TrainedModels\\20210401_B30_P1024_5F\\",useFilter=1,includeImage=0)
    
    fileList = [\
        "Results\Result_00\Result_B70_P8192_G4_TestSet_Karlstad.h5",\
        "Results\Result_00\Result_B70_P8192_G4_TestSet_Halsingborg.h5",\
        "Results\Result_00\Result_B70_P8192_G4_TestSet_Lund.h5",\
        "Results\Result_00\Result_B70_P8192_G4_TestSet_Norrkoping.h5",\
        "Results\Result_00\Result_B70_P8192_G4_TestSet_Nykoping.h5",\
        "Results\Result_00\Result_B70_P8192_G4_TestSet_Trollhattan.h5",\
        "Results\Result_00\Result_B70_P8192_G4_TestSet_Umea.h5",\
    ]

    analys_ALS([fileList[3]],"..\\TrainedModels\\20210401_B30_P1024_5F\\",useFilter=1,includeImage=1)



if __name__ == "__main__":
    main()

