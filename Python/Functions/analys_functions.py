import numpy as np
import pptk
import h5py
import importlib
import math
import os, os.path, sys 
import fnmatch
import shutil
from PIL import Image
import time

from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# ROC,Confusion matrix, Cohens kappa, Yoden's index
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt


def saveCoordinatesText(fileName,geo_coord,label_block,pred_label):
    ''' This function saves the coordinate of the missclassified tile blocks in a text file. '''

    # Get indecies of missclassified tile blocks.
    indexCoord = label_block != pred_label

    # Get the coordinates over the missclassified tile blocks.
    coordsToSave = geo_coord[indexCoord,:]

    # Write the coordinates in the file.
    np.savetxt(fileName,coordsToSave,fmt='%0.02d')



def rand_bin_array(K, N):
    ''' This function generates random binary array '''
    arr = np.zeros(N)
    arr[:K]  = 1
    np.random.shuffle(arr)
    return arr

def get_concat_h(im1, im2,im3):
    ''' This function merges the multiple images to one image in horizontal direction'''
    dst = Image.new('RGB', (im1.width + im2.width + im3.width, im1.height))
    im2width = im1.width + im1.width
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    dst.paste(im3, ( np.copy(im1.width) +im2.width, 0))
    return dst

def point_cloud_3D_view_plot(PointData,label_seg,pred_lab,block_num):
    ''' This function Plots the point cloud in 3D view with pptk'''
    # Plot the point cloud in 3D view with pptk 
    num_image = 3 # Print 3 images with 3 angel of view#import scikitplot as skplt
    img = []
    
    # PointDataSize = PointData.shape[0]
    # dataDim = PointData.shape[1]
    labelSize = label_seg.shape[0] # get data label size#import scikitplot as skplt
    # Define the color label
    color_label = np.zeros(labelSize)

    time.sleep(1)
    # Define the colors for point according to the confusion matrix
    for i in range(labelSize):
        if pred_lab[i]== label_seg[i]:           
            if pred_lab[i]==1:
                # TP red
                color_label[i]=1
            else:
                # TN green
                color_label[i]=0.5
        else:
            if pred_lab[i]==1:
                #FP yellow
                color_label[i]=0.7
            else:
                # FN Blue 
                color_label[i]=0
    
    time.sleep(2) # wait 1.5 seconds

    # Capture the png point image by pptk
    v = pptk.viewer(PointData,color_label) 
    v.color_map('jet', scale=[0, 1]) # define the color scale
    v.set(point_size=0.35) # define the point size

    lengthC0 = np.max(PointData[:,0]) - np.min(PointData[:,0])

    lengthC1 = np.max(PointData[:,1]) - np.min(PointData[:,1])

    radius = np.sqrt( lengthC0**2 + lengthC1**2 )*1.2

    v.set(r=radius) # define the radiun of the image view

    time.sleep(2)
    # Capture images with 3 angels
    for i in range(num_image):
        j = i 
        v.set(phi= i * np.pi/4)
        v.set(theta= i * np.pi/4)
        screenShotFileName = "image{}.png".format(j)
        v.capture(screenShotFileName)
 
    time.sleep(2) # wait 1.5 seconds
    v.close() # close pptk viewer
    # Open images saved in current file
    img1 = Image.open('image0.png')
    img2 = Image.open('image1.png')
    img3 = Image.open('image2.png')

    # Merge 3 images to one large image, then save it to folder 'Image'
    save_path ="Image"

    if not os.path.exists(save_path): os.mkdir(save_path)

    img_filename = 'Image_Block_%s.png' %(block_num)
    img_path = os.path.join(save_path,img_filename)
    image_pptk = get_concat_h(img1, img2,img3).save(img_path)
   # time.sleep(2) # wait 1.5 seconds

 #   return image_pptk

 
def ROC_AUC_analys_plot(y_actural_label,y_pred_label,filename):
    '''This function is used to Roc and AUC analys, function returns AUC value and plot/save an ROC AUC plot'''
    # Get the fpr(false positve rate), tpr(true positive rate) and threshold
    #fpr, tpr, threshold = metrics.roc_curve(y_actural_label, y_score)

    tn, fp, fn, tp = confusion_matrix(y_actural_label,y_pred_label).ravel()
    tpr = tp /(tp + fn)
    fpr = fp /(tn + fp)

    tpr_sum = np.sum(tpr)
    # Calculate the AUC( Area Under Curve)
    if np.isnan(tpr_sum):
        roc_auc = 0
        print("True positive rate is 0, very bad classifiser!")   
    else:
        roc_auc = metrics.auc(fpr, tpr)
    # plt plot 
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.show()
    plt.savefig(filename, format='png')
    plt.close()
    return (fpr,tpr,roc_auc)


def cohen_kappa_analys(y_actural_label,y_pred_label):
    '''This function generates a Cohen's kappa value k'''
    tn, fp, fn, tp = confusion_matrix(y_actural_label,y_pred_label).ravel()
    # Calculate the overall accuracy p_o
    p_o = (tp + tn)/(tp + tn + fp + fn)
    # Calculate probability for a random sample to be true positive:
    p_y = ((tp + fn)*(tp + fp))/((tp + tn + fp + fn)**2)
    # Calculate probability for a random sample to be true negative:
    p_n = ((fp + tn)*(fn + tn))/((tp + tn + fp + fn)**2)
    # Calculate probability for a random sample to be classified correctly:
    p_e = p_y + p_n
    # Calculate the cohen kappa value k:
    k = (p_o - p_e)/(1 - p_e)
    return k


def analys_score_methods(y_actural_label,y_pred_label):

    '''Function generates a Youden's index value J'''
    tn, fp, fn, tp = confusion_matrix(y_actural_label,y_pred_label).ravel()
    #tn, fp, fn, tp = confusion_matrix(y_actural_label,y_pred_label)
    print("TN:",tn,"TP:",tp,"FN:",fn,"FP:",fp)

    # Calculate the Yodens Index value J:
    if (tp+fn) == 0:
        J = 0
        precision_value = 0
        precision_value = tp/(tp + fp)
        recall_value = 0
        print("No bridge points, Recall is zero and Youden's value is not working.")   
    elif (tn + fp) == 0:
        J = 0
        precision_value = tp/(tp + fp)
        recall_value = tp/(tp + fn)
        print("Only bridge points, Youden's value is not working.")
    else:
        J = (tp/(tp + fn)) + (tn/(tn + fp)) - 1
        precision_value = tp/(tp + fp)
        recall_value = tp/(tp + fn)
        
    return J,precision_value,recall_value


    
def confusion_matrix_plot(y_actural_label,y_pred_label,filename):   
    '''Function generates the confusion matrix plot'''
    labels = ['0', '1'] # Set the class label, we have only 2 classes here
    tick_marks = np.array(range(len(labels))) + 0.5
    def plot_confusion_matrix(cm, title='Confusion Matrix', cmap = plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=90)
        plt.yticks(xlocations, labels)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
    # Get the confustion matrix cm
    cm = confusion_matrix(y_actural_label, y_pred_label)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12,8), dpi=120)
    #set the fontsize of label.Function generates the confusion matrix plot
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if (c > 0.01):
            plt.text(x_val, y_val, "%0.2f" %(c,), color='red', fontsize=7, va='center', ha='center')
    #offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)  
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    # plt.show()
    plt.savefig(filename, format='png')
    plt.close()


def learningCurvePlot(filename,savename):
    file1 = open(filename, 'r')
    Lines = file1.readlines()
    file1.close()

    validationAccuracy = []

    for i in range(len(Lines)-2):
        if(Lines[i].find("EVALUATION") >= 0):

            num = Lines[i+2].replace('eval accuracy: ','')

            validationAccuracy.append( np.double( num ) )

    plt.plot(validationAccuracy)
    plt.savefig(savename)






def CountLabelledBridges(coordinates,label_seg,pred_label_seg,geo_coord,thresH = 0.5):

    # Get dimensions of the block data.
    dims = coordinates.shape

    # Check if several tile blocks is included.
    if(len(dims) == 3 ):

        # Get number of blocks and points per block.
        nrBlocks = dims[0]
        nrPoints = dims[1]

        # Add the center coordinate to all the blocks to get orginal coordinates.
        for i in range(nrBlocks):
            coordinates[i,:,0:2] = coordinates[i,:,0:2] + np.flip( np.array([geo_coord[i,:],]*nrPoints) )

        # Merge the data to one dimension less.
        mergedCoord = coordinates.reshape(nrBlocks*nrPoints,3)
        mergedlabel_seg = label_seg.reshape(-1)
        mergedpred_label_seg = pred_label_seg.reshape(-1)

    elif(len(dims) == 2):
        # If there is only one tile block and the dimension is squeezed.
        nrBlock = 1
        mergedCoord = np.copy(coordinates)
        mergedlabel_seg = np.copy(label_seg)
        mergedpred_label_seg = np.copy(pred_label_seg)

    else:
        print("Wrong number of dimensions in 'coordinates'")
        assert(0)

    # Use DBSCAN to cluster labeled bridge points.
    clustering = DBSCAN(eps=4, min_samples=1).fit(mergedCoord[mergedlabel_seg == 1,0:2])

    # Create indecies for all points, zeros represent non bridge. 
    # Evey other is indicies for each bridge in the data.
    bridgeIndex = np.copy(mergedlabel_seg)
    bridgeIndex[mergedlabel_seg == 1] = clustering.labels_+ [1,]*len(clustering.labels_)

    # Debug tool
    # v = pptk.viewer(mergedCoord[mergedlabel_seg == 1,:], clustering.labels_)
    # v.set(point_size=0.35) # define the point size


    # Get number of labelled bridges.
    nrBridges = len(np.unique(clustering.labels_))
    # Count the number of bridges found.
    nrBridgesFound = 0

    tileBlocksWholeBridges = []
    predBridges = []
    labelBridges = []
    
    limitMargin = 20

    # Loop through all the bridges.
    for i in range(1,nrBridges):

        # Get the ratio of points found in the bridge.
        percentFound = np.sum(mergedpred_label_seg[bridgeIndex == i])/np.sum(bridgeIndex == i)

        # If the ratio of bridge points found is higher than the threshold, 
        # the bridge is labelled as found.
        if( thresH <= percentFound ):
            nrBridgesFound = nrBridgesFound + 1

        # Append one tile block over the bridge.

        # Max and min coordinate 0
        maxC0 = np.max( mergedCoord[bridgeIndex == i,0] ) + limitMargin
        minC0 = np.min( mergedCoord[bridgeIndex == i,0] ) - limitMargin

        # Max and min coordinate 1
        maxC1 = np.max( mergedCoord[bridgeIndex == i,1] ) + limitMargin
        minC1 = np.min( mergedCoord[bridgeIndex == i,1] ) - limitMargin

        # Get indecies of points within the limits.
        tempBridgeIndex = (minC0 < mergedCoord[:,0]) & (mergedCoord[:,0] < maxC0) & (minC1 < mergedCoord[:,1]) & (mergedCoord[:,1] < maxC1)

        # Save tileblock over the current bridge
        tileBlocksWholeBridges.append( mergedCoord[tempBridgeIndex,:] )

        labelBridges.append( mergedlabel_seg[tempBridgeIndex] )
        predBridges.append( mergedpred_label_seg[tempBridgeIndex] )

    # Debug
    # v = pptk.viewer( tileBlocksWholeBridges[2] )
    # v.set(point_size=0.35) # define the point size

    return nrBridgesFound,nrBridges,(tileBlocksWholeBridges,labelBridges,predBridges)
