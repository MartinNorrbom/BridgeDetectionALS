import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(ROOT_DIR, '../pointNET2/models'))
# sys.path.append(os.path.join(ROOT_DIR, '../pointNET2/utils'))

sys.path.insert(1, '../ML_Algorithms/pointNET2/part_seg')
sys.path.insert(1, '../ML_Algorithms/pointNET2/utils')
sys.path.insert(1, '../Functions')

import accessDataFiles
import analysis_ALS



# Specifies which point features that will be used during training.
# Leave empty for just XYZ. Write "return_number" to add number of returns and "intensity" to add intensity.

pointFeatures = ["intensity"] #["intensity","return_number"] #

logPath = '../TrainedModels/step_2_B60_P4096_E400/' #step_1_xyz+intensity_4F/' #step_1_xyz+RN_4F/' #step_1_xyz_3F/'


# Defines parameters for PointNet.
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
#parser.add_argument('--model', default='pointnet2_part_seg', help='Model name [default: pointnet2_part_seg]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=4096, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default=logPath+'model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--res_dir', default='Result', help='Result folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')


if(len(pointFeatures) == 0):
    parser.add_argument('--model', default='pointnet2_part_seg_3Features', help='Model name [default: model]')
elif(len(pointFeatures) == 1):
    parser.add_argument('--model', default='pointnet2_part_seg_4Features', help='Model name [default: model]')
elif(len(pointFeatures) == 2):
    parser.add_argument('--model', default='pointnet2_part_seg_5Features', help='Model name [default: model]')
else:
    print("pointFeatures was specified incorrectly.")
    assert(0)


FLAGS = parser.parse_args()

# Store arguments above as varables.
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
RES_DIR = FLAGS.res_dir
if not os.path.exists(RES_DIR): os.mkdir(RES_DIR)


# Define the number of classes.
NUM_CLASSES = 2


# SHAPE_NAMES = [line.rstrip() for line in \
#     open(os.path.join(BASE_DIR, '../data/Lantmateriet/shape_names.txt'))]

# Import the data.
# TRAIN_FILES = accessDataFiles.getDataFiles( \
#     os.path.join(BASE_DIR, '../data/Lantmateriet/train_files.txt'))

TEST_FILES = accessDataFiles.getDataFiles(\
    os.path.join(BASE_DIR, '../data/Lantmateriet/test_files.txt'))

RES_FILES = []


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# Function to evaluate the training.
def evaluate(num_votes):
    # Do not train the model.(Don't think it is doing anything)
    is_training = False
    
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        loss = MODEL.get_loss(pred, labels_pl)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)


    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss}

    print("Evaluate one epoch.")
    eval_one_epoch(sess, ops, num_votes)

    analysis_ALS.analys_ALS(RES_FILES,logPath)

    

# Function to evaluate one epoch.
def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    # Loop through all the test files.
    for fn in range(len(TEST_FILES)):

        currentResFile = accessDataFiles.copyFile(TEST_FILES[fn], RES_DIR)
        RES_FILES.append(currentResFile)

        # Load the current evaluation file.
        current_data, current_label, current_label_seg = accessDataFiles.load_h5_F5(TEST_FILES[fn],pointFeatures)

        # Get points for the input layer.
        current_data = current_data[:,0:NUM_POINT,:]
        current_label_seg = np.squeeze(current_label_seg)
        
        pred_label = np.copy(current_label)
        pred_label_seg = np.copy(current_label_seg)
        weight_label_seg = np.copy(current_label_seg)
        
        # Get number of batches.
        file_size = current_data.shape[0]  # number of blocks
        num_batches = file_size // BATCH_SIZE

        # Loop through all batches.
        for batch_idx in range(num_batches):
            # To get the samples in the current batch.
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            cur_batch_size = end_idx - start_idx

                
            # Merge the inputs to one variable.
            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                            ops['labels_pl']: current_label_seg[start_idx:end_idx,:],
                            ops['is_training_pl']: is_training}

            # Predict the batch with PointNet2.
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                        feed_dict=feed_dict)

            # Get the number of correct predictions.
            pred_val = np.argmax(pred_val, 2)
            # print(np.sum(pred_val))

            pred_label_seg[start_idx:end_idx,:] = np.copy( pred_val )
            # print(np.sum(pred_label_seg))

            #print(pred_val.shape)

            for i in range(cur_batch_size):

                if (np.sum(pred_val[i,:]) >= 1):
                    pred_label[start_idx+i] = 1
                else:
                    pred_label[start_idx+i] = 0
            # Get the number of correct predications in the current batch.
            correct = np.sum(pred_val == current_label_seg[start_idx:end_idx,:])


            # Add the scores for all the batches.
            total_correct += correct
            total_seen += (BATCH_SIZE*NUM_POINT)
            loss_sum += loss_val
            over_acc = total_correct/np.float(total_seen)


        accessDataFiles.add_predictions_h5(currentResFile,pred_label,[],pred_label_seg)
    
        
    print("Accuracy_seg: " + str(over_acc) )
                


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)
