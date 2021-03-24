'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, '../ML_Algorithms/pointNET2/models/'))
sys.path.append(os.path.join(ROOT_DIR, '../ML_Algorithms/pointNET2/utils/'))
import tf_util

import accessDataFiles


# Define parameters for PointNet++
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name [default: pointnet2_cls_ssg]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=1, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

# Save parameters as variables.
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

# Import PointNet, from the folder models.
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


# Define decay for learning-rate
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


NUM_CLASSES = 2  # trainingSkogData

# Import the data.
TRAIN_FILES = accessDataFiles.getDataFiles( \
    os.path.join(BASE_DIR, '../data/Lantmateriet/train_files.txt'))


TEST_FILES = accessDataFiles.getDataFiles(\
    os.path.join(BASE_DIR, '../data/Lantmateriet/test_files.txt'))

# Function to write a log in textfile and print it in terminal.
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


# Function to get the current learning-rate.
def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        


# Function to get the current decay-value for the learning-rate.
def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

# Function to train the network.
def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            MODEL.get_loss(pred, labels_pl, end_points)
            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)
            for l in losses + [total_loss]:
                tf.summary.scalar(l.op.name, l)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_acc = -1

        # Train and evaluate every epoch.
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            #if epoch % 10 == 0:
            save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
            log_string("Model saved in file: %s" % save_path)

# Function to train one epoch.
def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0


    # Loop through all the training files.
    for fn in range(len(TRAIN_FILES)):

        # Load the current evaluation file.
        current_data, current_label = accessDataFiles.load_h5_F5(TRAIN_FILES[fn])
        # Get points for the input layer.
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        
        pred_label = np.copy(current_label)
        
        print(current_data.shape)
        
        # Get number of samples.
        file_size = current_data.shape[0]
        # Get number of batches.
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

    print( "*** Training %d ***",fn )
    print("Accuracy: " + str(total_correct/np.float(total_seen)) )


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

        # Load the current evaluation file.
        current_data, current_label = accessDataFiles.load_h5_F5(TEST_FILES[fn])
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

    print( "*** Validation ***" )    
    print("Accuracy: " + str(total_correct/np.float(total_seen)) )


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
