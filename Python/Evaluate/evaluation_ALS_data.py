import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys

# Added imageio to save images.
import imageio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, '../ML_Algorithms/pointNET2/models/'))
sys.path.append(os.path.join(ROOT_DIR, '../ML_Algorithms/pointNET2/utils/'))
import accessDataFiles
import analysis_ALS


# Defines parameters for PointNet.
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='../ML_Algorithms/pointNET2/log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--res_dir', default='Result', help='Result folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
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
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, '../data/Lantmateriet/shape_names.txt'))]


# Import the data.
TRAIN_FILES = accessDataFiles.getDataFiles( \
    os.path.join(BASE_DIR, '../data/Lantmateriet/train_files.txt'))
# TEST_FILES = accessDataFiles.getDataFiles(\
#     os.path.join(BASE_DIR, 'debugFile.txt'))

TEST_FILES = accessDataFiles.getDataFiles(\
    os.path.join(BASE_DIR, '../data/Lantmateriet/test_files.txt'))

RES_FILES = []



# Function to evaluate the training.
def evaluate(num_votes):
    # Do not train the model.(Don't think it is doing anything)
    is_training = False
    
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        loss = MODEL.get_loss(pred, labels_pl, end_points)
        
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

    print("Evaluate on epoch.")
    eval_one_epoch(sess, ops, num_votes)

    analysis_ALS.analys_ALS(RES_FILES)
    



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
        current_data, current_label = accessDataFiles.loadDataFile(TEST_FILES[fn])
        # Get points for the input layer.
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        
        pred_label = np.copy(current_label)
        
        print(current_data.shape)
        
        # Get number of batches.
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        # Print number of samples.
        print(file_size)
        
        # Loop through all batches.
        for batch_idx in range(num_batches):
            # To get the samples in the current batch.
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            cur_batch_size = end_idx - start_idx

                
            # Merge the inputs to one variable.
            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                            ops['labels_pl']: current_label[start_idx:end_idx],
                            ops['is_training_pl']: is_training}

            # Predict the batch with PointNet.
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                        feed_dict=feed_dict)

            # Get the number of correct predictions.
            pred_val = np.argmax(pred_val, 1)
            pred_label[start_idx:end_idx] = pred_val

            # Get the number of correct predications in the current batch.
            correct = np.sum(pred_val == current_label[start_idx:end_idx])

            # Add the scores for all the batches.
            total_correct += correct
            total_seen += cur_batch_size
            loss_sum += (loss_val*BATCH_SIZE)

            # Save the results in a h5 file.
            #for i in range(start_idx, end_idx):

        accessDataFiles.add_predictions_h5(currentResFile,pred_label)
        
    
        
    print("Accuracy: " + str(total_correct/np.float(total_seen)) )
                





if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)
