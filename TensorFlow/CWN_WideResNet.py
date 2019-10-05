# CNNs with Label Noise - code for the paper "The Resistance to Label Noise in K-NN and CNN Depends on its Concentration" by Amnon Drory, Oria Ratzon, Shai Avidan and Raja Giryes
# 
# MIT License
# 
# Copyright (c) 2019 Amnon Drory
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import tensorflow as tf
import CodeSnapshots
import six

CodeSnapshots.snapshot_this_file('CWN_WideResNet')

def build_wide_resnet(images, P):
    """building the inference model of ResNet"""
    
    filters = [16,160,320,640]
    num_residual_units = 4
    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    
    with tf.variable_scope('init'):
        x = conv('init_conv', images, 3, filters[0], stride_arr(1))
        
    with tf.variable_scope('unit_1_0'):
        x = _residual(x, filters[1], stride_arr(strides[0]), activate_before_residual[0], P)
    for i in six.moves.range(1, num_residual_units):
        with tf.variable_scope('unit_1_%d' % i):
            x = _residual(x, filters[1], stride_arr(1), False, P)

    with tf.variable_scope('unit_2_0'):
        x = _residual(x, filters[2], stride_arr(strides[1]), activate_before_residual[1], P)
    for i in six.moves.range(1, num_residual_units):
        with tf.variable_scope('unit_2_%d' % i):
            x = _residual(x, filters[2], stride_arr(1), False, P)

    with tf.variable_scope('unit_3_0'):
        x = _residual(x, filters[3], stride_arr(strides[2]), activate_before_residual[2], P)
    for i in six.moves.range(1, num_residual_units):
        with tf.variable_scope('unit_3_%d' % i):
            x = _residual(x, filters[3], stride_arr(1), False, P)

    return unit_last(x, P)

def unit_last(x, P):
    """Implementing the final unit of the resnet"""
    with tf.variable_scope('unit_last'):
        x = activation(x, P)
        x = global_avg_pool(x)
        feature = x
        logits = fully_connected(feature, P['num_classes'])
        return logits, feature

def _residual(x, out_filter, stride, activate_before_residual, P):
    """Residual unit with 2 sub layers."""
    in_filter = x.get_shape().as_list()[3]
    if activate_before_residual:
        with tf.variable_scope('shared_activation'):
            x = activation(x, P)
            orig_x = x
    else:
        with tf.variable_scope('residual_only_activation'):
            orig_x = x
            x = activation(x, P)

    with tf.variable_scope('sub1'):
        x = conv('conv1', x, 3, out_filter, stride)

    with tf.variable_scope('sub2'):
        x = activation(x, P)
        x = conv('conv2', x, 3, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
        if in_filter != out_filter:
            orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
            orig_x = tf.pad(
                orig_x, [[0, 0], [0, 0], [0, 0],
                         [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
        x += orig_x

    return x

def stride_arr(stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

def conv(name, x, filter_size, out_filters, strides, padding='SAME'):
    """Convolution."""
    with tf.variable_scope(name):
        in_filters = x.get_shape().as_list()[3]
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable(
            'DW', [filter_size, filter_size, in_filters, out_filters],
            tf.float32, initializer=tf.random_normal_initializer(
            stddev=np.sqrt(2.0/n)))
        conv_out = tf.nn.conv2d(x, kernel, strides, padding=padding)
        return conv_out

def global_avg_pool(x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

def fully_connected(x, out_dim, name='fully_connected'):
    """FullyConnected layer for final output."""
    with tf.variable_scope(name):
        x_shape = x.get_shape().as_list()
        input_dim = np.prod(x_shape[1:])
        x = tf.reshape(x, [-1, input_dim])
        w = tf.get_variable(
            'DW', [input_dim, out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

def activation(x, P):
    x = tf.layers.batch_normalization(
        x,
        training=P['batchnorm_training'],
        momentum=P['batchnorm_momentum'],
        beta_initializer=tf.constant_initializer(0.1),
        gamma_regularizer=tf.contrib.layers.l2_regularizer(P['l2_reg'])
    )

    x = tf.nn.relu(x)
    return x