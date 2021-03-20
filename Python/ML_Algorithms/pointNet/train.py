import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util

# Defines parameters for PointNet.
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 2]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 4]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

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

# Import PointNet, from the folder models.
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

# Defines max number of points and classes
MAX_NUM_POINT = 1024
NUM_CLASSES = 1

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# Write the paths text-files with the selected data. 
# In our case the selected .h5 files that contains the training and test set.
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/Lantmateriet/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/Lantmateriet/test_files.txt'))


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

# Function to get the current decay-value to weight the learning-rate.
def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

# Function for the training of PointNet.
def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            #correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))

            # Changed for binary classifcation.
            correct = tf.equal(tf.to_int64(tf.greater(pred,0)), tf.to_int64(labels_pl))

            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

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
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        # Loop through all epoch.
        for epoch in range(MAX_EPOCH):
            # Print the current epoch.
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk. Save variable every 10:th epoch
            # if epoch % 10 == 0:
            save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
            log_string("Model saved in file: %s" % save_path)


# Function to train one epoch.
def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    # Indicate that model should be trained.
    is_training = True
    
    # Shuffle train files.
    # Randomize the order of reading the files in each epoch.
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)
    
    # Loop through all training files.
    for fn in range(len(TRAIN_FILES)):
        # Print/count the number of files during the epoch.
        log_string('----' + str(fn) + '-----')
        # Read the training data and labels in the current file.
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        # Collect input points for PointNet, from each sample it should be "NUM_POINT".
        current_data = current_data[:,0:NUM_POINT,:]
        # Randomize the samples in the training data.
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
        # Decrease dimensions in array, ex (1,3,1) to (3,)
        current_label = np.squeeze(current_label)
        
        # Get the size of the file.
        file_size = current_data.shape[0]
        # Flor devision to get the number of batches.
        num_batches = file_size // BATCH_SIZE
        
        total_correct = 0
        total_seen = 0
        loss_sum = 0
       
        # Train each batch.
        for batch_idx in range(num_batches):
            # To get the samples in the current batch.
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            # Augment batched point clouds by rotation and jittering
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            feed_dict = {ops['pointclouds_pl']: jittered_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,}
            
            # Train one batch.
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            
            # Log the training.
            train_writer.add_summary(summary, step)

            # Get classify the object as the output node with heighest weight.
            #pred_val = np.argmax(pred_val, 1)
            # Changed for binary output.

            print("Output weights: "+str(pred_val))
            pred_val = pred_val > 0
            
            
            print("Prediction: "+str(pred_val))
            print("Labeled: "+str(current_label[start_idx:end_idx]))
            correct = np.sum(pred_val == current_label[start_idx:end_idx])

            # Store the scores from all batches.
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val
        
        # Print the scores from the current training file.
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    # Indicate that model should not be trained.
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    # Loop through all test/validation files.
    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '-----')
        # Load the data from the test file.
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        
        # get number of batches per epoch.
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        for batch_idx in range(num_batches):
            # To get the samples in the current batch.
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE


            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            
            # Validate the current batch.
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)

            # Get the number of correct predictions.
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            
            # Store the scores from all batches.
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val*BATCH_SIZE)

            # Get accuracy per class.
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
            
    # Print the validation scores.
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
         


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
