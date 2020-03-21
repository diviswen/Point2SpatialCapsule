"""
    PointNet++ Model for point clouds classification
"""

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, './utils'))
sys.path.append(os.path.join(BASE_DIR, './external/structural_losses'))
import tf_approxmatch
import tf_nndistance
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module_msg, netVLAD
from capsnet import CapsLayer, squashing
from train import FLAGS
import loupe as lp
import tensorflow.contrib.slim as slim
M_PLUS = .9
M_MINUS = 0.1
LAMBDA_VAL = 0.5
epsilon = 1e-9
if FLAGS.modelnet10:
    NUM_CLASSES = 10
else:
    NUM_CLASSES = 40
def placeholder_inputs(batch_size, num_point):
    if FLAGS.normal:
        pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    else:
        pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, label, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = None
    ppc = []

    l0_xyz, l0_points = pointnet_sa_module_msg(l0_xyz[:,:,0:3], l0_points, 512, 
        radius_list=[0.1,0.2,0.4,0.8], 
        nsample_list=[8,16,32,64], 
        mlp_list=[[32,32,64], [32,32,64], [32,32,64], [32,32,64]], 
        is_training=is_training, 
        bn_decay=bn_decay,
        knn=True, 
        scope='layer1', 
        use_nchw=False, 
        rearrange=True)
    l0_xyz, l0_points = pointnet_sa_module_msg(l0_xyz[:,:,0:3], l0_points, 256, 
        radius_list=[0.1,0.2,0.4,0.8], 
        nsample_list=[8,16,32,64], 
        mlp_list=[[64,64,128], [64,64,128], [64,64,128], [64,64,128]], 
        is_training=is_training, 
        bn_decay=bn_decay,
        knn=True, 
        scope='layer0', 
        use_nchw=False, 
        rearrange=True)
    
    l0_points = tf.reshape(l0_points, [batch_size, 512, 256])
    print(l0_points.get_shape)
    with tf.variable_scope('netVLAD_feats'):
        l2_points = netVLAD(l0_points, 64, is_training=is_training, bn=True, bn_decay=bn_decay)
    with tf.variable_scope('netVLAD_xyz'):
        l2_xyz = netVLAD(l0_xyz, 64, is_training=is_training, bn=True, bn_decay=bn_decay)
    l2_points = tf.concat([l2_points, l2_xyz], axis=-1)
    l2_points = tf.contrib.layers.fully_connected(inputs=l2_points, num_outputs=256, scope='NetVLAD_fc',biases_initializer=None)
    l2_points = slim.batch_norm(l2_points,
              center=True,
              scale=True,
              is_training=is_training,
              scope="cluster_bn", fused=False, decay=bn_decay)

    ppc = tf.reshape(l2_points, (batch_size, 1, 1024, 16, 1))

    ppc = squashing(ppc)
    ppc = tf.squeeze(ppc)

    with tf.variable_scope('DigitCaps_Layers_1'):
        digitCaps = CapsLayer(input_number=1024, output_number=NUM_CLASSES, vec_length=16, out_length=32, layer_type='FC')
        caps = digitCaps(ppc, is_training=is_training, bn=False)


    one_hot_label = tf.one_hot(label, depth=NUM_CLASSES, axis=1, dtype=tf.float32)
    masked_v = tf.matmul(tf.squeeze(caps),
                      tf.reshape(one_hot_label, (-1, NUM_CLASSES, 1)), transpose_a=True)
    with tf.variable_scope('Reconstruct'):

        v_j = tf.reshape(masked_v, shape=(batch_size, -1))
        fc = tf.contrib.layers.fully_connected(inputs=v_j, num_outputs=128, scope="fc1", biases_initializer=None)
        fc = slim.batch_norm(fc,
          center=True,
          scale=True,
          is_training=is_training,
          scope="fc1", fused=False, decay=bn_decay)
        
        fc = tf.contrib.layers.fully_connected(inputs=fc, num_outputs=256, scope="fc2", biases_initializer=None)
        fc = slim.batch_norm(fc,
          center=True,
          scale=True,
          is_training=is_training,
          scope="fc2", fused=False, decay=bn_decay)

        fc = tf.contrib.layers.fully_connected(inputs=fc, num_outputs=512, scope="fc3", biases_initializer=None)
        
        fc = slim.batch_norm(fc,
          center=True,
          scale=True,
          is_training=is_training,
          scope="fc3", fused=False, decay=bn_decay)
        
        fc = tf.contrib.layers.fully_connected(inputs=fc, num_outputs=1024*3, scope="out", biases_initializer=None, activation_fn=None)
        net = tf.reshape(fc, (batch_size, 1024, 1, 3))

        reconstruct = tf.squeeze(net)
    end_points['reconstruct'] = tf.reshape(reconstruct, shape=(batch_size, 1024, 3))
    return caps, end_points


def get_loss(caps, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    batch_size = caps.get_shape()[0].value
    one_hot_label = tf.one_hot(label, depth=NUM_CLASSES, axis=1, dtype=tf.float32)

    masked_v = tf.matmul(tf.squeeze(caps),
                      tf.reshape(one_hot_label, (-1, NUM_CLASSES, 1)), transpose_a=True)
    v_length = tf.sqrt(tf.reduce_sum(tf.square(caps), axis=2, keep_dims=True)
                    + epsilon)
    batch = tf.get_collection('batch')[0]
    if FLAGS.spread:
        # spread loss
        v_length = tf.reshape(v_length, shape=[-1, 1, NUM_CLASSES])
        one_hot_label = tf.expand_dims(one_hot_label, axis=2)
        at = tf.matmul(v_length, one_hot_label)
        """Paper eq(5)."""
        m = tf.minimum(0.9, 0.4+batch*(0.9-0.2)/25000)
        loss = tf.square(tf.maximum(0., m - (at - v_length)))
        loss = tf.matmul(loss, 1. - one_hot_label)
        loss = tf.reduce_mean(loss)
    else:
        # 1. margin_loss
        M_PLUS_ = tf.minimum(1., 0.8+0.2*batch/10000)
        M_MINUS_ = tf.maximum(0., 0.2-0.2*batch/10000)
        max_l = tf.square(tf.maximum(0., M_PLUS - v_length))
        max_r = tf.square(tf.maximum(0., v_length - M_MINUS))
        assert max_r.get_shape() == [batch_size, NUM_CLASSES, 1, 1]
        # reshape: [batch_size, NUM_CLASSES, 1, 1] => [batch_size, NUM_CLASSES]
        max_l = tf.reshape(max_l, shape=(batch_size, -1))
        max_r = tf.reshape(max_r, shape=(batch_size, -1))
        T_c = one_hot_label
        # element-wise multiply, [batch_size, NUM_CLASSES]
        L_c = T_c * max_l + LAMBDA_VAL * (1 - T_c) * max_r
        loss = tf.reduce_mean(tf.reduce_mean(L_c, axis=1))

    tf.summary.scalar('classify loss', loss)
    tf.add_to_collection('losses', loss)

    match = tf_approxmatch.approx_match(end_points['reconstruct'], end_points['l0_xyz'])
    reconstruct_loss = 0.0001 * tf.reduce_mean(tf_approxmatch.match_cost(end_points['reconstruct'], end_points['l0_xyz'], match))

    tf.add_to_collection('losses', reconstruct_loss)
    return loss, reconstruct_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        output, _ = get_model(inputs, tf.constant(True))
        print(output)
