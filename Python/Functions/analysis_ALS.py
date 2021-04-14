import os
import sys
import numpy as np

import accessDataFiles
import analys_functions

import filterFunctions

def analys_ALS(filename,logpath, useFilter = 0,includeImage = 0):


    data,label_block,label_seg,pred_label,pred_label_seg,geo_coord = \
    accessDataFiles.load_h5_analys_data(filename[0])


    if(includeImage == 1):
        for i in range(data.shape[0]):

            pred_label_seg[i,:] = filterFunctions.pointFilter(data[i,:,0:3],pred_label_seg[i,:],10,20,3)
            
            tempVar = np.sum(pred_label_seg[i,:])
            if(tempVar == 0):
                pred_label[i] = 0

    nrBridgesFound,nrBridges,bridgeInfo = analys_functions.CountLabelledBridges(data,label_seg,pred_label_seg,geo_coord)
    print("Number of bridges found: " + str(nrBridgesFound))
    print("Total number of bridges: " + str(nrBridges))

    if( includeImage == 1 ):
        for i in range( len(bridgeInfo[0]) ):
            analys_functions.point_cloud_3D_view_plot( bridgeInfo[0][i],bridgeInfo[1][i],bridgeInfo[2][i], i )



    ##################### Get Learning curve #######################

    analys_functions.learningCurvePlot( logpath+'log_train.txt','learning_curve.png')


    ##################### Analysis For Block Classification #######################

    # Create a text file that contains the coordinates that have been missclassificated.
    if( len(geo_coord) > 0 ):
        analys_functions.saveCoordinatesText("coordinates.txt",geo_coord,label_block,pred_label)

    # Create an image over the confusion matrix.
    analys_functions.confusion_matrix_plot(label_block,pred_label,"ConfusionMatrix_Classification.png")

    # Calculate Youden's J statistics.
    youdenScore_Class, precision_Class, recall_Class = analys_functions.analys_score_methods(label_block,pred_label)

    # Calculate Cohen's kappa value.
   # cohenScore_Class = analys_functions.cohen_kappa_analys(label_block,pred_label)

    # Print Youden's J statistics and Cohen's kappa.
    print('Youdens index value for blocks: '+str(youdenScore_Class))
    print('Precision value for blocks: '+str(precision_Class))
    print('Recall value for blocks: '+str(recall_Class))



    ###############################################################################

    # Needs to be tested.

    ###################### Analysis For Point Segmentation ########################

    # Check if prediction for segmentation is available.
    if(  len(pred_label_seg) > 0 ):


        label_seg_total = np.asarray(label_seg).reshape(-1)
        pred_label_seg_total = np.asarray(pred_label_seg).reshape(-1)

        youdenScore_Seg, precision_seg, recall_seg = analys_functions.analys_score_methods(label_seg_total,pred_label_seg_total)

        #cohenScore_Seg = analys_functions.cohen_kappa_analys(label_seg_total,pred_label_seg_total)


        analys_functions.confusion_matrix_plot(label_seg_total,pred_label_seg_total,"ConfusionMatrix_Segmentaion.png")

        # Print Youden's J statistics and Cohen's kappa.
        print('Youdens index value for points: '+str(youdenScore_Seg))
        print('Precision value for points: '+str(precision_seg))
        print('Recall value for points: '+str(recall_seg))



        # Create images
        if (includeImage):
            for i in range(len(label_block)):
                if label_block[i] != pred_label[i]:

                    analys_functions.point_cloud_3D_view_plot( data[i,:,:],label_seg[i,:],pred_label_seg[i,:], i )


    ###############################################################################





def main():
    #analys_ALS(["../Test/TestResults/step_2_B60_P4096/400Epochs/Karlstad/Result/Result_B60_P4096_Karlstad_Test_Set.h5"],"../TrainedModels/step_2_B60_P4096_E400/")

    analys_ALS(["../Test/TestResults/step_2_B60_P4096/400Epochs/OnlyBridge/Result/Result_B60_P4096_OnlyBridge_Test_Set.h5"],"../TrainedModels/step_2_B60_P4096_E400/")



if __name__ == "__main__":
    main()




