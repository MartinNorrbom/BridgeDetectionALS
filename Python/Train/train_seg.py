import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(ROOT_DIR, '../ML_Algorithms/pointNET2/part_seg'))
# sys.path.append(os.path.join(ROOT_DIR, '../ML_Algorithms/pointNET2/utils'))

sys.path.insert(1, '../ML_Algorithms/pointNET2/part_seg')
sys.path.insert(1, '../ML_Algorithms/pointNET2/utils')
sys.path.insert(1, '../Functions')

#import provider
#import tf_util
#import part_dataset_all_normal
import accessDataFiles

# Specifies which point features that will be used during training.
# Leave empty for just XYZ. Write "return_number" to add number of returns and "intensity" to add intensity.
pointFeatures = ["return_number", "intensity"] #["intensity"] #


# Defines parameters for PointNet++
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='../TrainedModels/log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')



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

EPOCH_CNT = 0

# Store arguments above as varables.
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')

# Create an log dir if it doesn't exists
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


# Define the number of classes.
NUM_CLASSES = 2 



# Import data for train and test files:
TRAIN_FILES = accessDataFiles.getDataFiles( \
    os.path.join(BASE_DIR, '../data/Lantmateriet/train_files.txt'))

TEST_FILES = accessDataFiles.getDataFiles(\
    os.path.join(BASE_DIR, '../data/Lantmateriet/validation_files.txt'))

RES_FILES = []

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
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
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_acc = -1

        maxValidationAccuracy = 0.0


        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            validationAccuracy = eval_one_epoch(sess, ops, test_writer)

            print(maxValidationAccuracy)
            print(validationAccuracy)

            # Save the trained model with highest validation accuracy.
            if (maxValidationAccuracy < validationAccuracy):

                maxValidationAccuracy = np.copy(validationAccuracy)

                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True   

    # Shuffle train samples
    # train_idxs = np.arange(0, len(TRAIN_DATASET))
    # np.random.shuffle(train_idxs)
    # num_batches = len(TRAIN_DATASET)/BATCH_SIZE
    
    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0



    # Loop through all the test files.
    for fn in range(len(TRAIN_FILES)):


        # Load the current training file.
        current_data, current_label, current_label_seg = accessDataFiles.load_h5_F5(TRAIN_FILES[fn],pointFeatures)
        print(str(TRAIN_FILES[fn]))

        # Get points for the input layer.
        current_data = current_data[:,0:NUM_POINT,:]
        current_label_seg = np.squeeze(current_label_seg)

        # Get number of batches.
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE


        for batch_idx in range(int(num_batches)):

            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                            ops['labels_pl']: current_label_seg[start_idx:end_idx,:],
                            ops['is_training_pl']: is_training}


            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)

            train_writer.add_summary(summary, step)

            pred_val = np.argmax(pred_val, 2)
            correct = np.sum(pred_val == current_label_seg[start_idx:end_idx,:])
            total_correct += correct
            total_seen += (BATCH_SIZE*NUM_POINT)
            loss_sum += loss_val

            # Check the performance for every 100th batch
            if (batch_idx+1)%100 == 0:
                log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
                log_string('mean loss: %f' % (loss_sum / 100))
                log_string('accuracy: %f' % (total_correct / float(total_seen)))
                total_correct = 0
                total_seen = 0
                loss_sum = 0
            

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    validationLength = 0
    total_correct = 0
    total_seen = 0
    loss_sum = 0

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    
    batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 5))
    batch_label = np.zeros((BATCH_SIZE, NUM_POINT)).astype(np.int32)

    # Loop through all the test files.
    for fn in range(len(TEST_FILES)):

        # Load the current evaluation file.
        current_data, current_label, current_label_seg = accessDataFiles.load_h5_F5(TEST_FILES[fn],pointFeatures)
        print(str(TEST_FILES[fn]))
        # Get points for the input layer.
        current_data = current_data[:,0:NUM_POINT,:]
        current_label_seg = np.squeeze(current_label_seg)

        # Get number of batches.
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        # Loop through all batches.
        for batch_idx in range(int(num_batches)):


            # To get the samples in the current batch.
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            cur_batch_size = end_idx - start_idx            
            # Merge the inputs to one variable.
            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                            ops['labels_pl']: current_label_seg[start_idx:end_idx,:],
                            ops['is_training_pl']: is_training}

            # Predict the batch with PointNet2.
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)

            test_writer.add_summary(summary, step)

            pred_val = np.argmax(pred_val, 2)  

            # Get the number of correct predications in the current batch.
            correct = np.sum(pred_val == current_label_seg[start_idx:end_idx,:])
            total_correct += correct
            total_seen += (BATCH_SIZE*NUM_POINT)
            loss_sum += loss_val

    # Print out the mean loss and overall accuracy
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/BATCH_SIZE)))

    validationAccuracy = (total_correct / float(total_seen))

    log_string('eval accuracy: %f'% validationAccuracy)

    EPOCH_CNT += 1
    return validationAccuracy


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
