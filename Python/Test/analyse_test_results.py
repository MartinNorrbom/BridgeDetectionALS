import os
import sys
import numpy as np

sys.path.insert(1, '../Functions')

import accessDataFiles
import analys_functions

import analysis_ALS
import filterFunctions
import training_functions
import pptk
from laspy.file import File


def main():

    # Path for the orginal LAZ files from Lantmäteriet.
    pathOldLabel = "/media/snaffe/TOSHIBA EXT/Exjobb/Saved_Laz_Files/"

    # Path for the Laz files only predicted by the algorithm that Lantmäteriet use. 
    pathOldPred = "/media/snaffe/TOSHIBA EXT/Exjobb/Old_Algorithm_Performans/laz_bro/"

    # Folder for the files saved after the analysis.
    sFolders = ["Results_Analysis/Overlap_00/","Results_Analysis/Overlap_25/","Results_Analysis/Overlap_50/"]

    # Analyse files that has not used voting.
    #analysis_ALS.analys_ALS(fileList_50,[],useFilter=0,includeImage=0,saveFolder=sFolders[2],saveFiltered=0)

    # Analyse files that has used voting, and set different parameter settings for the filter. 
    analysis_ALS.analys_ALS(fileList_50_Voted,[],useFilter=1,includeImage=0,saveFolder=sFolders[2],saveFiltered=0)

    # Visualise one point cloud over one city. 
    visulate_city(city = 0,showNonML = 0)
    



def visulate_city(city = 0,showNonML = 0):
    ''' This functions makes plots the whole point cloud over an selected city.\
    The input "city" is the city number and if "showNonML" is 1 it will also \
    include a plot from the algorithm that Lantmäteriet use.\
    '''


    # Get the h5 file for the specified city.
    data,label_block,label_seg,pred_label,pred_label_seg,geo_coord = \
        accessDataFiles.load_h5_analys_data(filesToAnalyseVisually[city])

    # Set the colour for each prediction. (Red = TP, Yellow = FP, Green = TN, Blue = FN)
    color = np.zeros(len(label_seg))
    color[ (label_seg == pred_label_seg) & (label_seg == 1)] = 1
    color[ (label_seg != pred_label_seg) & (label_seg == 1)] = 0.01
    color[ (label_seg != pred_label_seg) & (label_seg == 0)] = 0.7

    # Start the point cloud viewer.
    vF = pptk.viewer(data[color != 0], color[color != 0] )
    vF.set(point_size=0.35)             # Define the point size.
    vF.color_map('jet', scale=[0,1])    # Define the colour scale.


    # Visualization old algorithm, make sure that you have those files first...
    if(showNonML == 1):
        
        # Get LAZ file for the city.
        currentCityRes = oldAlgorithmResults[city]

        cityPoints = []
        cityPred = []
        cityLabel = []

        # Loop through all the files for the current city.
        for i in range( len(currentCityRes) ):

            # Read LAZ files
            predfile = File(pathOldPred+currentCityRes[i], mode='r')
            labelfile = File(pathOldLabel+currentCityRes[i], mode='r')

            # Append theom to the temporary list.
            cityPoints.append( np.transpose( np.array([predfile.x,predfile.y,predfile.z]) ) )
            cityPred.append( predfile.classification == 17 )
            cityLabel.append( labelfile.classification == 17 )

        # Merge files.
        oldPoints = np.concatenate( cityPoints )
        oldPred = np.concatenate( cityPred )
        oldLabel = np.concatenate( cityLabel )

        # Set the colour for each prediction. (Red = TP, Yellow = FP, Green = TN, Blue = FN)
        color = np.zeros(len(oldLabel))
        color[ (oldLabel == oldPred) & (oldLabel == 1)] = 1
        color[ (oldLabel != oldPred) & (oldLabel == 1)] = 0.01
        color[ (oldLabel != oldPred) & (oldLabel == 0)] = 0.7
        
        # Start the point cloud viewer.
        vO = pptk.viewer(oldPoints[color != 0], color[color != 0] )
        vO.set(point_size=0.35)             # Define the point size.
        vO.color_map('jet', scale=[0,1])    # Define the colour scale.
    



filesToAnalyseVisually = [\
    "Results_Analysis/Overlap_50/Final_Filter_h5/res_filtering0.h5",    \
    "Results_Analysis/Overlap_50/Final_Filter_h5/res_filtering1.h5",    \
    "Results_Analysis/Overlap_50/Final_Filter_h5/res_filtering2.h5",    \
    "Results_Analysis/Overlap_50/Final_Filter_h5/res_filtering3.h5",    \
    "Results_Analysis/Overlap_50/Final_Filter_h5/res_filtering4.h5",    \
    "Results_Analysis/Overlap_50/Final_Filter_h5/res_filtering5.h5",    \
    "Results_Analysis/Overlap_50/Final_Filter_h5/res_filtering6.h5"]

# Laz file names.
# Halsingborg, Karlstad, Lund, Norrkoping, Nykoping, Trollhattan, Umea
oldAlgorithmResults = [             \
[   "19A007_62100_3550_25.laz",     \
    "19A007_62100_3575_25.laz",     \
    "19A007_62150_3575_25.laz"],    \
[   "19B030_65825_4125_25.laz",     \
    "19B030_65825_4150_25.laz"],    \
[   "18A003_61725_3850_25.laz",     \
    "18A003_61725_3875_25.laz",     \
    "18A005_61750_3850_25.laz",     \
    "18A005_61750_3875_25.laz"],    \
[   "20C012_64925_5650_25.laz",     \
    "20C012_64925_5675_25.laz",     \
    "20C012_64950_5650_25.laz",     \
    "20C012_64950_5675_25.laz"],    \
[   "20C020_65125_6125_25.laz",     \
    "20C020_65125_6150_25.laz",     \
    "20C020_65150_6125_25.laz",     \
    "20C020_65150_6150_25.laz"],    \
[   "20B016_64625_3400_25.laz",     \
    "20B016_64625_3425_25.laz",     \
    "20B016_64650_3400_25.laz",     \
    "20B016_64650_3425_25.laz"],    \
[   "20F036_70850_7575_25.laz",     \
    "20F036_70875_7575_25.laz"]     \
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

fileList_50_Voted = [\
    ["Results_Analysis/Overlap_50/Final_Voting_h5/res_voting0.h5"],    \
    ["Results_Analysis/Overlap_50/Final_Voting_h5/res_voting1.h5"],    \
    ["Results_Analysis/Overlap_50/Final_Voting_h5/res_voting2.h5"],    \
    ["Results_Analysis/Overlap_50/Final_Voting_h5/res_voting3.h5"],    \
    ["Results_Analysis/Overlap_50/Final_Voting_h5/res_voting4.h5"],    \
    ["Results_Analysis/Overlap_50/Final_Voting_h5/res_voting5.h5"],    \
    ["Results_Analysis/Overlap_50/Final_Voting_h5/res_voting6.h5"]]


if __name__ == "__main__":
    main()