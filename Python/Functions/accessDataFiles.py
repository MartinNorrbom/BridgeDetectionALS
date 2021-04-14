import os
import sys
import numpy as np
import h5py
import shutil
import ntpath
import time


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename,'r')
    data = f['data'][:]
    label = f['label'][:]
    f.close()
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)


def load_h5_data_label_seg(filename):
    f = h5py.File(filename,'r')
    data = f['data'][:]
    label = f['label'][:]
    seg = f['label_seg'][:]
    f.close()
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)


def load_h5_F5(filename,pointFeatures):
    f = h5py.File(filename,'r')
    XYZ = f['data'][:]
    intens = f['intensity'][:]
    rNumber = f['return_number'][:]
    label = f['label'][:]
    seg = f['label_seg'][:]

    # Return the specified point features.
    if(len(pointFeatures)==0):
        data = np.copy(XYZ)
    elif(len(pointFeatures)==1):
        if(pointFeatures[0] == "return_number"):
            data = np.concatenate((XYZ,rNumber),axis=2)
        elif(pointFeatures[0] == "intensity"):
            data = np.concatenate((XYZ,intens),axis=2)
        else:
            print("Invalid pointFeatures.")
            assert(0)
    elif(len(pointFeatures)==2):
        data = np.concatenate((XYZ,intens,rNumber),axis=2)
    else:
        print("pointFeatures had an invalid size.")
        assert(0)

    f.close()
    return (data, label, seg)


def load_h5_analys_data(filename):
    f = h5py.File(filename,'r')
    
    listOfTypes = ['data','label','label_seg','pred_label','pred_label_seg','geo_coord']

    typesAvailable = np.zeros(len(listOfTypes))

    fileKeys = list(f.keys())

    for i in range(len(listOfTypes)):
        for cKey in fileKeys:
            if(listOfTypes[i] == cKey):
                typesAvailable[i]=1
                break
    
    if sum(typesAvailable[0:4]) < 4:
        print("The h5 file is not supported, at least one key of the types("+str(listOfTypes[0:4])+")")
    else:
        data = f[listOfTypes[0]][:]
        label = f[listOfTypes[1]][:]
        seg = f[listOfTypes[2]][:]
        pred_label = f[listOfTypes[3]][:]

    if typesAvailable[4]:
        pred_label_seg = f[listOfTypes[4]][:]
    else:
        pred_label_seg = []

    if typesAvailable[5]:
        geo_coord = f[listOfTypes[5]][:]
    else:
        geo_coord = []

    f.close()

    return (data,label,seg,pred_label,pred_label_seg,geo_coord)



def add_predictions_h5(filename,pred_label,*arg):
    '''This function adds prediction from the ML algorithm to the h5 file.'''
    hf = h5py.File(filename, 'a')
    hf.create_dataset('pred_label',data = pred_label)

    if (len(arg)>0):
        hf.create_dataset('CW_label',data = arg[0])

    if (len(arg)>1):
        hf.create_dataset('pred_label_seg',data = arg[1])

    if (len(arg)>2):
        hf.create_dataset('CW_label_seg',data = arg[2])

    hf.close()



def copyFile(filename,outputPath,nameToAdd="Result_"):
    '''This function copies an already existing file to a new place.'''

    # Get file name.
    head, tail = ntpath.split(filename)

    # Get destination path and name.
    dst_filename = os.path.join(outputPath, nameToAdd+tail)

    print("Copy file")
    # Copy file.
    shutil.copy(filename,dst_filename)

    # Wait until the whole file is copied.
    currentSize = -1

    fileSize = os.path.getsize(filename)

    while currentSize != fileSize:
        currentSize = os.path.getsize(dst_filename)
        time.sleep(1)
    print("Finnished copying")

    return dst_filename
