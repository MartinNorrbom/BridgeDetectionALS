import os
import sys
import numpy as np

import accessDataFiles
import analys_functions


def analys_ALS(filename):

    # INDICATE IF SEGMENTATION IMAGES SHOULD BE SAVED, SHOULD BE SET AS AN INPUT PARAMETER INSTEAD.
    includeImage = 1


    data,label_block,label_seg,pred_label,pred_label_seg,geo_coord = \
    accessDataFiles.load_h5_analys_data(filename[0])


    ##################### Analysis For Block Classification #######################

    # Create a text file that contains the coordinates that have been missclassificated.
    if( len(geo_coord) > 0 ):
        analys_functions.saveCoordinatesText("coordinates.txt",geo_coord,label_block,pred_label)

    # NEED WEIGHT OF OUTPUT FOR RoC-CURVE
    analys_functions.ROC_AUC_analys_plot(label_block,pred_label,"RoC_Classification.png")

    # Create an image over the confusion matrix.
    analys_functions.confusion_matrix_plot(label_block,pred_label,"ConfusionMatrix_Classification.png")

    # Calculate Youden's J statistics.
    youdenScore_Class = analys_functions.youdens_index_analys(label_block,pred_label)

    # Calculate Cohen's kappa value.
    cohenScore_Class = analys_functions.cohen_kappa_analys(label_block,pred_label)

    # Print Youden's J statistics and Cohen's kappa.
    print('Youdens for block: '+str(youdenScore_Class))
    print('Cohens for block: '+str(cohenScore_Class))

    ###############################################################################

    # Needs to be tested.

    ###################### Analysis For Point Segmentation ########################

    # Check if prediction for segmentation is available.
    if(  len(pred_label_seg) > 0 ):

        # Create images
        if (includeImage):
            for i in range(len(label_block)):
                if label_block[i] != pred_label[i]:

                    analys_functions.point_cloud_3D_view_plot( data[i,:,:],label_seg[i,:],pred_label_seg[i,:], i )


        tot_label_seg = np.copy(label_seg).flatten('C')

        tot_pred_label_seg = np.copy(pred_label_seg).flatten('C')

        # NEED WEIGHT OF OUTPUT FOR RoC-CURVE
        analys_functions.ROC_AUC_analys_plot(tot_label_seg,tot_pred_label_seg,"RoC_Segmentation.png")

        youdenScore_Seg = analys_functions.youdens_index_analys(tot_label_seg,tot_pred_label_seg)

        cohenScore_Seg = analys_functions.cohen_kappa_analys(tot_label_seg,tot_pred_label_seg)

        analys_functions.confusion_matrix_plot(tot_label_seg,tot_pred_label_seg,"ConfusionMatrix_Segmentaion.png")


    ###############################################################################








