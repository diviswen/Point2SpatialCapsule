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
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import modelnet_dataset
import modelnet_h5_dataset
import time
from retrieval import retrival_results
import copy
import scipy
from scipy.spatial.distance import euclidean, cosine
import pc_util
epsilon = 1e-9
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='p2sc', help='Model name [default: pointnet2_cls_ssg]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=1000, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=20, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information')
parser.add_argument('--spread', action='store_true', default=False, help='Whether to use spread loss')
parser.add_argument('--weight_decay', action='store_true', default=False, help='Whether to use weight_decay')
parser.add_argument('--augment', action='store_true', default=False, help='Whether to use augmentation')
parser.add_argument('--modelnet10', action='store_true', default=False, help='Whether to use modelnet10')
parser.add_argument('--iter_routing', type=int, default=1, help='The number of iterations of dynamic routing')

FLAGS = parser.parse_args()

EPOCH_CNT = 0
HEADER = 'ply\nformat ascii 1.0\nelement vertex 1024\nproperty double x\nproperty double y\nproperty double z\nend_header\n'
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
MODEL_NAME = 'pretrained'
WEIGHT_DECAY = FLAGS.weight_decay
WEIGHT_DECAY_RATE = 1e-5
AUGMENT = FLAGS.augment

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir

FEATURE_DIR = os.path.join(LOG_DIR, MODEL_NAME, 'features')
if not os.path.exists(FEATURE_DIR): os.mkdir(FEATURE_DIR)
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

global ACC_GLOBAL
ACC_GLOBAL = 0.93

if FLAGS.modelnet10:
    NUM_CLASSES = 10
else:
    NUM_CLASSES = 40

# Shapenet official train/test split
if FLAGS.modelnet10 or FLAGS.normal:
    assert(NUM_POINT<=10000)
    DATA_PATH = os.path.join(ROOT_DIR, 'data/modelnet40_normal_resampled')
    TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, modelnet10=FLAGS.modelnet10, batch_size=BATCH_SIZE)
    TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, modelnet10=FLAGS.modelnet10, batch_size=BATCH_SIZE)
else:
    assert(NUM_POINT<=2048)
    TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=True)
    TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    #print(out_str)

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

def eval():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            pointclouds_pl_ORIGIN = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            tf.add_to_collection('batch', batch)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)
            tf.add_to_collection('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, labels_pl, bn_decay=bn_decay)

            end_points['l0_xyz'] = pointclouds_pl_ORIGIN
            margin_loss, reconstruct_loss = MODEL.get_loss(pred, labels_pl, end_points)
            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)
            for l in losses + [total_loss]:
                tf.summary.scalar(l.op.name, l)

            v_length = tf.sqrt(tf.reduce_sum(tf.square(tf.squeeze(pred)), axis=2, keep_dims=False))

            correct = tf.equal(tf.argmax(v_length, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            u_v_list = []
            for r_iter in range(FLAGS.iter_routing):
                u_v = tf.get_collection('u_v_%d'%(r_iter))[0]
                u_v_list.append(tf.reduce_mean(tf.abs(u_v)))
            u_v_list = tf.stack(u_v_list, axis=0)
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'eval'+'/test'+'/tensorboard'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'pointclouds_pl_ORIGIN': pointclouds_pl_ORIGIN,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': v_length,
               'loss': total_loss,
               'merged': merged,
               'step': batch,
               'end_points': end_points,
               'margin_loss': margin_loss, 
               'reconstruct_loss': reconstruct_loss,
               'mse': u_v_list,
               'reconstruction': end_points['reconstruct']}
        saver.restore(sess, os.path.join(LOG_DIR, MODEL_NAME,'model.ckpt'))
        eval_one_epoch(sess, ops, test_writer) 
        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
    cur_batch_data_ORIGIN = np.zeros((BATCH_SIZE,NUM_POINT,3))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    shape_ious = []
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    pred_set = []
    label_set = []
    reconstruction_set = []
    original_set = []
    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label
        cur_batch_data_ORIGIN[0:bsize,...] = batch_data[:,:,0:3]


        num_votes=1
        batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES)) # score for classes
        for vote_idx in range(num_votes):
            # Shuffle point order to achieve different farthest samplings
            shuffled_indices = np.arange(NUM_POINT)
            np.random.shuffle(shuffled_indices)
            if FLAGS.normal:
                #rotated_data = provider.rotate_point_cloud_by_angle_with_normal(cur_batch_data[:, shuffled_indices, :],
                #    vote_idx/float(num_votes) * np.pi * 2)
                #rotated_data = provider.random_point_dropout(cur_batch_data[:, shuffled_indices, :], max_dropout_ratio=0.4)
                #xyz_data = provider.random_scale_point_cloud(cur_batch_data[:, :, 0:3], scale_low=0.9, scale_high=1.1)
                #cur_batch_data[:, :, 0:3] = xyz_data
                rotated_data = cur_batch_data[:, shuffled_indices, :]
            else:
                #rotated_data = provider.rotate_point_cloud_by_angle(cur_batch_data[:, shuffled_indices, :],
                #    vote_idx/float(num_votes) * np.pi * 2)
                #rotated_data =cur_batch_data[:, shuffled_indices, :]
                rotated_data = provider.random_scale_point_cloud(cur_batch_data[:, shuffled_indices, :], scale_low=0.9, scale_high=1.2)
            feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                         ops['labels_pl']: cur_batch_label,
                         ops['is_training_pl']: is_training,
                         ops['pointclouds_pl_ORIGIN']: cur_batch_data_ORIGIN}
            summary, step, loss_val, pred_val, reconstruction = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred'], ops['reconstruction']], feed_dict=feed_dict)
            batch_pred_sum += pred_val
        pred_val = batch_pred_sum/float(num_votes)
        #sess.run(ops['reset_b_IJ'])
        pred_set.append(copy.deepcopy(pred_val))
        label_set.append(copy.deepcopy(cur_batch_label))
        reconstruction_set.append(copy.deepcopy(reconstruction))
        original_set.append(copy.deepcopy(batch_data))
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        batch_idx += 1
        for i in range(0, bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)

    print(str(datetime.now()))
    print('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    print('MODEL: %s'%(MODEL_NAME))
    print('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    print('eval accuracy: %f'% (total_correct / float(total_seen)))
    print('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))

    pred_set = np.concatenate(pred_set, axis=0)
    label_set = np.concatenate(label_set, axis=0)
    reconstruction_set = np.concatenate(reconstruction_set, axis=0)
    original_set = np.concatenate(original_set, axis=0)
    
    np.save(os.path.join(FEATURE_DIR, 'test_features.npy'), pred_set)
    np.save(os.path.join(FEATURE_DIR, 'test_labels.npy'), label_set)
    np.save(os.path.join(FEATURE_DIR, 'reconstruction.npy'), reconstruction_set)
    mAPs, _ = retrival_results(os.path.join(FEATURE_DIR, "test_features.npy"),
                 os.path.join(FEATURE_DIR, "test_labels.npy"),
                 os.path.join(FEATURE_DIR, "test_features.npy"),
                 os.path.join(FEATURE_DIR, "test_labels.npy"),
                 save_dir=FEATURE_DIR)
    print('eval test2test mAP: %.5f'%(mAPs[0]))
    '''
    EPOCH_CNT += 1
    
    for i in range(40):
        file_original = '%d_original.jpg' % (i)
        file_reconstruct = '%d_reconstruct.jpg' % (i)
        file_original = os.path.join(FEATURE_DIR, file_original)
        file_reconstruct = os.path.join(FEATURE_DIR, file_reconstruct)
        reconstruct_img = pc_util.point_cloud_three_views(np.squeeze(reconstruction_set[i*20, :, :]))
        original_img = pc_util.point_cloud_three_views(np.squeeze(original_set[i*20, :, :]))
        scipy.misc.imsave(file_reconstruct, reconstruct_img)
        scipy.misc.imsave(file_original, original_img)
        
        f_xyz_original = open(os.path.join(FEATURE_DIR, '%d_original.ply' % (i)),'w')
        f_xyz_original.write(HEADER)
        f_xyz_reconstruct = open(os.path.join(FEATURE_DIR, '%d_reconstruct.ply' % (i)),'w')
        f_xyz_reconstruct.write(HEADER)
        for j in range(1024):
            xyz = np.squeeze(original_set[i*30, :, :])
            f_xyz_original.write('%f %f %f\n'%(xyz[j][0],xyz[j][1],xyz[j][2]))

            xyz = np.squeeze(reconstruction_set[i*30, :, :])
            f_xyz_reconstruct.write('%f %f %f\n'%(xyz[j][0],xyz[j][1],xyz[j][2]))
        f_xyz_original.close()
        f_xyz_reconstruct.close()
    '''
    TEST_DATASET.reset()
    return total_correct/float(total_seen)


if __name__ == "__main__":
    print('pid: %s'%(str(os.getpid())))
    eval()
