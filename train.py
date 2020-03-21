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
epsilon = 1e-9
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='p2sc', help='Model name [default: pointnet2_cls_ssg]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
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

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
MODEL_NAME = '%s-%s'%(FLAGS.model, time.strftime("%m%d_%H%M%S", time.localtime()))
WEIGHT_DECAY = FLAGS.weight_decay
WEIGHT_DECAY_RATE = 1e-5
AUGMENT = FLAGS.augment

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
if not os.path.exists(os.path.join(LOG_DIR, MODEL_NAME)): os.mkdir(os.path.join(LOG_DIR, MODEL_NAME))

FEATURE_DIR = os.path.join(LOG_DIR, MODEL_NAME, 'features')
if not os.path.exists(FEATURE_DIR): os.mkdir(FEATURE_DIR)
os.system('cp %s %s/%s/%s.py' % (MODEL_FILE, LOG_DIR, MODEL_NAME, MODEL_NAME)) # bkp of model def
os.system('cp train_vlad_ablation.py %s/%s/train_vlad_ablation_%s.py' % (LOG_DIR, MODEL_NAME, MODEL_NAME)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, MODEL_NAME, 'log_%s.csv' %(MODEL_NAME)), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

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
if FLAGS.modelnet10:
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

def train():
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

            print "--- Get training operator"
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9)

            u_v_list = []
            for r_iter in range(FLAGS.iter_routing):
                u_v = tf.get_collection('u_v_%d'%(r_iter))[0]
                u_v_list.append(tf.reduce_mean(tf.abs(u_v)))
            u_v_list = tf.stack(u_v_list, axis=0)

            b_IJ = tf.constant(np.zeros([16, 1024, 10, 1, 1], dtype=np.float32))
            reset_b_IJ = None#tf.assign(tf.get_collection('b_IJ')[0], b_IJ)

            # weight decay operation
            weights_var = tf.trainable_variables()
            l2_loss = WEIGHT_DECAY_RATE * tf.add_n([tf.nn.l2_loss(v) for v in weights_var])
            sgd = tf.train.GradientDescentOptimizer(learning_rate=learning_rate*1000)
            decay_op = sgd.minimize(l2_loss)
            if WEIGHT_DECAY:
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, decay_op)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(total_loss, global_step=batch)

            with tf.control_dependencies([train_op]):
                updata_b_IJ = tf.assign(tf.get_collection('b_IJ')[0], tf.reduce_mean(tf.get_collection('b_IJ_out')[0], axis=0, keep_dims=True))
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
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, MODEL_NAME+'/train'+'/tensorboard'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, MODEL_NAME+'/test'+'/tensorboard'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'pointclouds_pl_ORIGIN': pointclouds_pl_ORIGIN,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': v_length,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points,
               'margin_loss': margin_loss, 
               'reconstruct_loss': reconstruct_loss,
               'mse': u_v_list,
               'reset_b_IJ': reset_b_IJ,
               'l2_loss': l2_loss}

        best_acc = -1
        for epoch in range(MAX_EPOCH):
            print('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            train_one_epoch(sess, ops, train_writer)
            acc = eval_one_epoch(sess, ops, test_writer) 

            # Save the variables to disk.
            if acc > best_acc:
                best_acc = acc
                save_path = saver.save(sess, os.path.join(LOG_DIR, MODEL_NAME, 'model.ckpt'))
                print("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    print(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TRAIN_DATASET.num_channel()))
    cur_batch_data_ORIGIN = np.zeros((BATCH_SIZE,NUM_POINT,3))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    reconstruct_sum =0.
    margin_sum = 0.
    mse_sum = np.array([0. for _ in range(FLAGS.iter_routing)])
    l2_loss_sum = 0.
    loss_sum = 0.
    batch_idx = 0
    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=AUGMENT)
        bsize = batch_data.shape[0]

        cur_batch_data_ORIGIN[0:bsize,...] = copy.deepcopy(batch_data[:,:,0:3])
        batch_data = provider.random_point_dropout(batch_data, max_dropout_ratio=0.5)
        #cur_batch_data_ORIGIN[0:bsize,...] = batch_data[:,:,0:3]
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training,
                     ops['pointclouds_pl_ORIGIN']: cur_batch_data_ORIGIN}
        u_v_list, summary, step, _, loss_val, pred_val, margin_loss, reconstruct_loss, l2_loss = sess.run([ops['mse'], ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred'], ops['margin_loss'], ops['reconstruct_loss'], ops['l2_loss']], feed_dict=feed_dict)
        #sess.run(ops['reset_b_IJ'])
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        mse_sum += u_v_list
        reconstruct_sum += reconstruct_loss
        margin_sum += margin_loss
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        l2_loss_sum += l2_loss
        if (batch_idx+1)%50 == 0:
            mse_str = str([round(x, 3) for x in (mse_sum/50).tolist()])
            print(' ---- batch: %03d ----' % (batch_idx+1))
            print('mean loss: %.5f | accuracy: %.5f | margin_loss: %.5f | rec_loss: %.5f | mse: %s | l2_loss: %.4f | step: %d' \
                %(loss_sum/50, total_correct/float(total_seen), margin_sum/50, reconstruct_sum/50, mse_str, l2_loss_sum/50, int(step)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            reconstruct_sum =0.
            margin_sum = 0.
            mse_sum *= 0.
            l2_loss_sum = 0.
        batch_idx += 1

    TRAIN_DATASET.reset()
        
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
    
    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label
        cur_batch_data_ORIGIN[0:bsize,...] = batch_data[:,:,0:3]

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training,
                     ops['pointclouds_pl_ORIGIN']: cur_batch_data_ORIGIN}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        #sess.run(ops['reset_b_IJ'])
        test_writer.add_summary(summary, step)
        pred_set.append(copy.deepcopy(pred_val))
        label_set.append(copy.deepcopy(cur_batch_label))
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
    np.save(os.path.join(FEATURE_DIR, 'test_features.npy'), pred_set)
    np.save(os.path.join(FEATURE_DIR, 'test_labels.npy'), label_set)
    mAPs = [0.]
    '''
    mAPs, _ = retrival_results(os.path.join(FEATURE_DIR, "test_features.npy"),
                 os.path.join(FEATURE_DIR, "test_labels.npy"),
                 os.path.join(FEATURE_DIR, "test_features.npy"),
                 os.path.join(FEATURE_DIR, "test_labels.npy"),
                 save_dir=FEATURE_DIR)
    print('eval test2test mAP: %.5f'%(mAPs[0]))
    '''
    log_string('%s,%03d,%.4f,%.4f,%.4f,%.4f'%(str(datetime.now()),EPOCH_CNT, 
                                        loss_sum / float(batch_idx), 
                                        total_correct / float(total_seen), 
                                        np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)),
                                        mAPs[0]
                                        )
                )

    EPOCH_CNT += 1

    TEST_DATASET.reset()
    return total_correct/float(total_seen)


if __name__ == "__main__":
    print('pid: %s'%(str(os.getpid())))
    log_string('time,epoch,Eval mean loss,Eval accuracy,eval avg class acc')
    train()
    LOG_FOUT.close()
