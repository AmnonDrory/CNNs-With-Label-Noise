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
import os
import sys
# select which GPU to work on (read from first command line argument) (this code has to come before "import tensorflow")
if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]  # select the gpu to use

import tensorflow as tf
import datetime
import glob
from shutil import copyfileobj
import traceback
import platform
from scipy.signal import medfilt

from CWN_WideResNet import build_wide_resnet
import CWN_env
import CodeSnapshots
from CWN_data_loader import CWN_data_loader

# sets of noise levels used for each experiment
NOISE_LEVELS = {
    'uniform': [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.95, 0.99],
    'flip': [0, 0.2, 0.3, 0.35, 0.4,0.43, 0.46, 0.48, 0.5, 0.52, 0.54, 0.57, 0.6, 0.65, 0.7, 0.8],
    'generalCorruptionMatrix': [0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    'locally_concentrated': 1.0/np.array([10, 8, 6, 5, 4, 3, 2]),
    'none': [0.0]
}

P = {  # parameters to controls the training, see CnnWithNoise.train() below
    'experiment':
        {
            'repeats': 1, # how many times to try each noise level
        },
    'data':
        {
            'dataset': 'cifar10', # 'mnist', 'cifar10'
            'normalization': None, # specified ahead in default_normalization
            'train_labels_file': None, # None - default file. Otherwise, specify path to file.
            'noise_type': 'locally_concentrated', # 'none', 'uniform', 'flip', 'generalCorruptionMatrix',  'locally_concentrated'
            'noisy_validation': False,
        },
    'net':
        {
            'main_structure': 'allconv', # 'lenet', 'allconv', 'wide_resnet'
            'batch_size': 256,  # number of samples per minibatch
            'batchnorm_momentum': 0.99,
            'with_feature': False
        },
    'train':
        {
            'epochs': 60,
        },
    'report':
        {
            # how often to report accuracy/loss for different sets. set to np.nan to not report.
            'train': 10,  # how often to print out loss/accuracy on minibatch
            'validation': 10,  # how often to calculate and print the loss on the entire validation and test sets
            'whole_train_set': np.nan
        },
    'solver':
        {
            'learning_rate': 0.007,  # initial
            'epochs_till_decrease': 30,  # every K epochs decrease by FACTOR,
            'decrease_factor': 2,  # where K='epochs_till_decrease', FACTOR='decrease_factor'
            'max_times_to_decrease': np.inf,
            'l2_reg': 10 ** -7,  # a.k.a weight-decay
        },
    'early_stop':
        {
            'initial_iters_to_skip': 1000,
            'iters_to_smooth': 1,
            'min_vals_to_smooth': 1
        },
    'saver':
        {
            'interval': np.inf,  # save out checkpoint of network every 'interval' iterations
            'restore': False  # if true, initialize the network from latest checkpoint
        },
    'predict':
        {
            'features_after_relu': False,
            'output_data_for_probs': False,
            'best':
                {
                    'output_predictions': False,
                    'output_features': False,
                    'output_weights': False
                },
            'general':
                {
                    'output_predictions': False,
                    'output_features': False,
                    'when': range(0, 20000, 2000)
                }
        }
}

default_normalization = {'cifar10': 'per_pixel_mean_std',
                         'mnist': '0.0_to_1.0'}

P['data']['normalization'] = default_normalization[P['data']['dataset']]

class CnnWithNoise():
            
    def parse_command_line_arguments(self):
        # range of run numbers can be set from command line
        self.num_screens = None
        self.screen_ind = None
        if len(sys.argv) > 3:
            self.screen_ind = int(sys.argv[2])
            self.num_screens = int(sys.argv[3])
            # argv[2] is surbrange ind (screen ind), starts from 1
            # argv[3] is total number of subranges (screens)
        self.logs_list_name = None
        if len(sys.argv) > 4:
            self.logs_list_name = sys.argv[4]

    # ===============================================================
    #                       NETWORK
    # ===============================================================
    def main_network_structure_wide_resnet(self):
        images_shape = list(self.D.data['train']['images'].shape)
        images_shape[0] = None
        image_wid = images_shape[1]

        images = tf.placeholder(tf.float32, shape=images_shape, name='images')

        if len(images_shape) == 3:
            reshaped_images = tf.reshape(images, [-1, image_wid, image_wid,1])
        else:
            reshaped_images = images

        WRN_params = {
            'batchnorm_training': self.batchnorm_training,
            'batchnorm_momentum': self.P['net']['batchnorm_momentum'],
            'l2_reg': self.P['solver']['l2_reg'],
            'num_classes': self.D.num_classes
        }

        logits, feature = build_wide_resnet(reshaped_images, WRN_params)

        tf.add_to_collection('PREDICTION_LAYERS', feature)
        self.meaningful_names[feature.name] = 'feature'

        self.predict = logits # storing in class member so we can reference it in train()

        self.report_free_parameters("predict")

        return logits

    def main_network_structure_allconv(self):
        """
        Defines the main network. 

        Remark: Does NOT include:
         * calculation of loss
         * definition of solver
         * auxiliary nodes used for statistics, saving checkpoints, etc.
         see build_net() for those. 
        """

        # based on the network in '/home/ad/PycharmProjects/reference/Colorization/EfrosColorization/models/colorization_deploy_v1_b.prototxt'
        images_shape = list(self.D.data['train']['images'].shape)
        images_shape[0] = None
        image_wid = images_shape[1]

        images = tf.placeholder(tf.float32, shape=images_shape, name='images')

        if len(images_shape) == 3:
            reshaped_images = tf.reshape(images, [-1, image_wid, image_wid,1])
        else:
            reshaped_images = images

        cur_image_wid = image_wid

        conv_1_1 = self.conv_block('1_1', reshaped_images, 3, 96)
        conv_1_2 = self.conv_block('1_2', conv_1_1, 3, 96)
        conv_1_3 = self.conv_block('1_3', conv_1_2, 3, 96, subsample=True)
        cur_image_wid /= 2

        conv_2_1 = self.conv_block('2_1', conv_1_3, 3, 192)
        conv_2_2 = self.conv_block('2_2', conv_2_1, 3, 192)
        conv_2_3 = self.conv_block('2_3', conv_2_2, 3, 192, subsample=True)
        cur_image_wid /= 2

        conv_3_1 = self.conv_block('3_1', conv_2_3, 3, 192, valid_only=True)
        cur_image_wid -= 2
        conv_3_2 = self.conv_block('3_2', conv_3_1, 1, 192)

        if 'with_feature' in self.P['net']['main_structure']:

            FEATURE_SIZE = 256
            conv_3_3 = self.conv_block('3_3', conv_3_2, 1, FEATURE_SIZE)

            global_averaging_res = tf.nn.avg_pool(conv_3_3,
                                                  ksize=[1, cur_image_wid, cur_image_wid, 1],
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID')

            feature_with_relu, _ = self.dense_block(global_averaging_res, FEATURE_SIZE, 'feature', output_feature=True)

            predict, _ = self.dense_block(feature_with_relu, self.D.num_classes, 'predict', with_ReLU=False)

        else:

            conv_3_3 = self.conv_block('3_3', conv_3_2, 1, self.D.num_classes)

            predict_raw = tf.nn.avg_pool(conv_3_3,
                                     ksize=[1,cur_image_wid,cur_image_wid,1],
                                     strides=[1,1,1,1],
                                     padding='VALID')

            predict = tf.reshape(predict_raw, [-1, self.D.num_classes])

        self.predict = predict # storing in class member so we can reference it in train()

        self.report_free_parameters("predict")

        return predict

    def main_network_structure_lenet(self):
        """
        Defines the main network. 

        Remark: Does NOT include:
         * calculation of loss
         * definition of solver
         * auxiliary nodes used for statistics, saving checkpoints, etc.
         see build_net() for those. 
        """

        self.feature_size = 256

        images_shape = list(self.D.data['train']['images'].shape)
        images_shape[0] = None
        image_wid = images_shape[1]

        images = tf.placeholder(tf.float32, shape=images_shape, name='images')
        if len(images_shape) == 3:
            reshaped_images = tf.reshape(images, [-1, image_wid, image_wid,1])
        else:
            reshaped_images = images

        conv_1 = self.conv_block('1', reshaped_images, 5, 20)
        conv_2 = self.conv_block('2', conv_1, 5, 20)
        pool1 = self.pool(conv_2, 'pool1')
        conv_3 = self.conv_block('3', pool1, 5, 50)
        conv_4 = self.conv_block('4', conv_3, 5, 50)
        pool2 = self.pool(conv_4, 'pool2')
        fc1, _ = self.dense_block(pool2, self.feature_size, 1, output_feature=True)
        fc2, _ = self.dense_block(fc1, self.D.num_classes, 2, with_ReLU=False)

        predict = tf.reshape(fc2, [-1, self.D.num_classes]) # may not actually be necessary

        self.predict = predict # storing in class member so we can reference it in train()

        self.report_free_parameters("predict")

        return predict

    def define_one_loss(self, prefix, labels, predict, zero_weight=False, is_inlier=None):

        loss_per_sample = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,
            logits=predict,
            name=prefix + '_softmax_cross_entropy_loss'
        )

        loss_per_sample = tf.reshape(loss_per_sample,[-1])

        predict_per_sample = tf.argmax(predict, axis=1)
        label_per_sample = tf.argmax(labels, axis=1)
        is_correct_per_sample = tf.equal(predict_per_sample, label_per_sample)
        is_correct_per_sample_float32 = tf.cast(is_correct_per_sample, tf.float32)
        accuracy = tf.reduce_mean(is_correct_per_sample_float32)
        accuracy.weights = 0
        tf.add_to_collection(tf.GraphKeys.LOSSES, accuracy)
        self.meaningful_names[accuracy.name] = prefix + '_accuracy'

        if is_inlier is not None:
            mask = {'corrupt':tf.logical_not(is_inlier), 'unchanged': is_inlier}
            for subset in ['corrupt', 'unchanged']:
                is_correct_per_sample_subset = tf.boolean_mask(is_correct_per_sample,mask[subset])
                num_correct_subset = tf.reduce_sum(tf.cast(is_correct_per_sample_subset,tf.float32))
                num_total_subset = tf.reduce_sum(tf.cast(mask[subset],tf.float32))
                tf.add_to_collection("LOSSES_ON_CORRUPT_SAMPLES", num_correct_subset)
                tf.add_to_collection("LOSSES_ON_CORRUPT_SAMPLES", num_total_subset)
                self.meaningful_names[num_correct_subset.name] = 'num_correct_' + subset
                self.meaningful_names[num_total_subset.name] = 'num_total_' + subset

        loss = tf.reduce_mean(loss_per_sample)
        if zero_weight:
            loss.weights = 0
        else:
            loss.weights = 1
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
        self.meaningful_names[loss.name] = prefix + '_loss'

    def define_loss(self, predict):

        noisy_labels = tf.placeholder(tf.float32, shape=[None, self.D.num_classes], name='noisy_labels')
        true_labels = tf.placeholder(tf.float32, shape=[None, self.D.num_classes], name='true_labels')
        is_inlier = tf.placeholder(tf.bool, shape=[None], name='is_inlier')
        self.define_one_loss('noisy', noisy_labels, predict, is_inlier=is_inlier)
        self.define_one_loss('true', true_labels, predict, zero_weight=True)

        # calculate weighted sum of losses:
        total_loss = 0.0
        for cur_loss in (tf.get_collection(tf.GraphKeys.LOSSES) +
                             tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)):
            try:
                total_loss += (cur_loss.weights * cur_loss)
            except:  # weights is not a field, assume weight of 1
                total_loss += cur_loss

        return total_loss

    def build_net(self):
        """
        Builds the network
        """

        # -------------
        # Place holders
        # --------------
        self.batchnorm_training = tf.placeholder(shape=(), dtype=tf.bool, name='batchnorm_training')

        # -------------
        # Main network
        # -------------
        if self.P['net']['main_structure'] == 'lenet':
            predict = self.main_network_structure_lenet()
        elif 'allconv' in self.P['net']['main_structure']:
            predict = self.main_network_structure_allconv()
        elif self.P['net']['main_structure'] == 'wide_resnet':
            predict = self.main_network_structure_wide_resnet()
        else:
            assert False, "unknown main structure: " + self.P['net']['main_structure']

        # ----
        # Loss
        # -----
        total_loss = self.define_loss(predict)

        # ------
        # Solver
        # -------
        self.learning_rate = tf.placeholder(shape=(), dtype=tf.float32)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  # necessary for batchnorm
            optimizer_base = tf.train.AdamOptimizer(self.learning_rate)
            optimizer_base.minimize(total_loss)

        total_free_parameters = self.count_free_parameters()
        self.print_and_log('Total Free Parameters: %d' % total_free_parameters)

        self.saver = tf.train.Saver(name='saver')

    def conv_block(self, name,  inp, wid, out_channels, subsample=False, with_relu=True, valid_only=False, with_batchnorm=True):
        """
        A block of layers:
        convolutional 3x3,
        then ReLU,
        then max pool with step 2,

        :param inp: input tensor
        :param out_channels: number of output channel of the conv layer
        :param layer_index:
        :return:
        """
        name = str(name)
        conv_lyr = self.conv(inp, wid, out_channels, subsample, valid_only, name="conv" + name)

        self.report_free_parameters("conv" + name)

        if with_batchnorm:
            batchnorm_lyr = tf.layers.batch_normalization(
                conv_lyr,
                momentum=self.P['net']['batchnorm_momentum'],
                training=self.batchnorm_training,
                beta_initializer=tf.constant_initializer(0.1),
                gamma_regularizer=tf.contrib.layers.l2_regularizer(self.P['solver']['l2_reg']),
                name="batchnorm" + name)
        else:
            batchnorm_lyr = conv_lyr

        self.report_free_parameters("batchnorm" + name)

        if with_relu:
            relu_lyr = tf.nn.relu(batchnorm_lyr, name="relu" + name)
        else:
            relu_lyr = batchnorm_lyr

        return relu_lyr

    def conv(self, inp, wid, out_channels, subsample=False, valid_only=False, name=None):
        """
        conv layer

        :param inp: input tensor
        :param out_channels: number of output channel (a.k.a filters)
        :param name:
        :return: a 3x3 conv layer
        """
        if subsample:
            strides = 2
        else:
            strides = 1

        if valid_only:
            padding = 'valid'
        else:
            padding = 'same'

        res = tf.layers.conv2d(
            inputs=inp,
            filters=out_channels,
            kernel_size=[wid, wid],
            strides=strides,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.P['solver']['l2_reg']),
            use_bias=False,
            padding=padding,
            name=name,
            activation=None)

        return res

    def flatten(self, inp, name=None):
        """

        :param inp: input tensor
        :param name:
        :return: flattened version of tensor (all elements of each sample are a vector)
        """
        inp_shape = inp.shape.as_list()
        sample_size = np.prod(inp_shape[1:])
        flat_inp = tf.reshape(inp, [-1, sample_size], name=name)
        return flat_inp

    def dense_block(self, inp, out_channels, name, with_ReLU=True, output_feature=False, output_weights=False):
        """
        
        :param inp:
        :param out_channels:
        :param layer_index:
        :return:
        """

        name = str(name)
        with_batchnorm = with_ReLU

        flat_inp = self.flatten(inp)

        dense_lyr = tf.layers.dense(
            inputs=flat_inp,
            units=out_channels,
            use_bias=not with_batchnorm,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.P['solver']['l2_reg']),
            name="dense" + name,
            activation=None)

        feature = dense_lyr

        if output_feature and not self.P['predict']['features_after_relu']:
            tf.add_to_collection('PREDICTION_LAYERS', feature)
            self.meaningful_names[feature.name] = 'fc' + name

        self.report_free_parameters("dense" + name)

        if with_batchnorm:
            batchnorm_lyr = tf.layers.batch_normalization(
                dense_lyr,
                momentum=self.P['net']['batchnorm_momentum'],
                training=self.batchnorm_training,
                beta_initializer=tf.constant_initializer(0.1),
                gamma_regularizer=tf.contrib.layers.l2_regularizer(self.P['solver']['l2_reg']),
                name="batchnorm_dense_" + name)

            if output_weights:
                for fld in ['gamma', 'beta', 'moving_mean','moving_variance']:
                    fld_name = "batchnorm_dense_" + name + '/' + fld + ':0'
                    fld_tensor = self.graph.get_tensor_by_name(fld_name)
                    tf.add_to_collection('PREDICTION_WEIGHTS', fld_tensor)
                    self.meaningful_names[fld_name] = 'fc' + name + '.' + fld

        else:
            batchnorm_lyr = dense_lyr

            if output_weights:
                bias_name = "dense" + name + '/bias:0'
                bias = self.graph.get_tensor_by_name(bias_name)
                tf.add_to_collection('PREDICTION_WEIGHTS', bias)
                self.meaningful_names[bias_name] = 'fc' + name + '.bias'

        self.report_free_parameters("batchnorm_dense_" + name)

        if with_ReLU:
            relu_lyr = tf.nn.relu(batchnorm_lyr, name="relu" + name)
        else:
            relu_lyr = batchnorm_lyr

        if output_feature and self.P['predict']['features_after_relu']:
            tf.add_to_collection('PREDICTION_LAYERS', relu_lyr)
            self.meaningful_names[relu_lyr.name] = 'fc_relu.' + name

        if output_weights:
            W_name = "dense" + name + '/kernel:0'
            W = self.graph.get_tensor_by_name(W_name)
            tf.add_to_collection('PREDICTION_WEIGHTS', W)
            self.meaningful_names[W_name] = 'fc' + name + '.W'

        return relu_lyr, feature

    def dense(self, inp, out_channels, is_output_layer=False, name=None):
        """
        Fully-connected (=dense) layer

        :param inp: input tensor
        :param out_channels: number of output channel (a.k.a units)
        :param is_output_layer: don't add batchnorm, and don't follow by relu
        :param name:
        :return: a Fully-connected layer
        """
        flat_inp = self.flatten(inp)

        d = tf.layers.dense(
            inputs=flat_inp,
            units=out_channels,
            use_bias=True,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.P['solver']['l2_reg']),
            bias_initializer=tf.constant_initializer(0.1),
            bias_regularizer=tf.contrib.layers.l2_regularizer(self.P['solver']['l2_reg']),
            name=name,
            activation=None)

        return d

    def pool(self, inp, name=None):
        """
        Max pool layer with step 2

        :param inp: input tensor
        :param name:
        :return: Max pool layer
        """
        return tf.layers.max_pooling2d(inp, 3, 2, padding="same", name=name)

    def report_free_parameters(self, layer_name):
        """

        :return:
        """
        total_free_parameters = self.count_free_parameters()

        try:
            cur_free_parameters = total_free_parameters - self.total_free_parameters
        except AttributeError as E:
            if str(E) == "CnnWithNoise instance has no attribute 'total_free_parameters'":
                cur_free_parameters = total_free_parameters
            else:
                raise E

        self.total_free_parameters = total_free_parameters
        self.print_and_log('%s free parameters: %d' % (layer_name, cur_free_parameters))

    def count_free_parameters(self):
        """

        :return:
        """
        total_free_parameters = 0
        for tnsr in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            total_free_parameters += np.prod(np.array(tnsr.get_shape().as_list()))
        return total_free_parameters

    def load_data(self):
        self.D = CWN_data_loader(self.P, self.noise_level, self.print_and_log,
                                 **self.P['data'])

        # output noisy labels
        for role in self.D.data.keys():
            if (self.D.data[role]['noisy_labels']==self.D.data[role]['true_labels']).all():
                continue
            FN = self.predict_dir[role] + 'noisy_labels.bin'
            lbls = self.D.data[role]['noisy_labels'].argmax(axis=1)
            with open(FN, 'wb') as fid:
                lbls.astype('uint8').tofile(fid)

        E = self.P['train']['epochs']
        N = self.D.data['train']['num_samples']
        B = self.P['net']['batch_size']
        if B > N:
            self.print_and_log(
                "Requested batch size of %d is larger than num_samples %d. Effective batch size will be %d" % (B,N,N))
            B = N
            self.P['net']['batch_size'] = B
        self.iters_per_epoch = np.ceil(float(N)/B)
        self.num_iters = int(E*self.iters_per_epoch)
        self.print_and_log(
            "%d epochs * %f iterations per epoch = %d iterations" % (E, self.iters_per_epoch, self.num_iters))

        if self.P['predict']['output_data_for_probs']:
            self.output_data_for_probs('train')
            self.output_data_for_probs('test')
            self.output_data_for_probs('validation')
    # =================================================================
    #                          INITIALIZATION
    # =================================================================
    def __init__(self, P, repeat_num=0):
        """
        Initializations        
        """
        self.parse_command_line_arguments()
        self.P = P
        self.meaningful_names = {}

        self.initialize_log_file()

        self.randomize_parameters(repeat_num)

        self.print_and_log_simple("\n")
        self.print_and_log_simple("Command Line Arguments: " + str(sys.argv))
        for k1 in self.P.keys():
                for k2 in self.P[k1].keys():
                    if isinstance(self.P[k1][k2], dict):
                        for k3 in self.P[k1][k2].keys():
                            self.print_and_log_simple(('P[%s][%s][%s] = ' % (k1, k2,k3)) + str(P[k1][k2][k3]))
                    else:
                        self.print_and_log_simple(('P[%s][%s] = ' % (k1, k2)) + str(P[k1][k2]))

        # create directory for checkpoints:
        if not os.path.isdir(CWN_env.checkpoints_dir):
            os.makedirs(CWN_env.checkpoints_dir)

        self.predict_dir = {}
        self.predict_dir['train'] = CWN_env.predictions_dir + 'train/' + str(self.run_number) + '/'
        self.predict_dir['validation'] = self.predict_dir['train'].replace('train','validation')
        self.predict_dir['test'] = self.predict_dir['train'].replace('train','test')

        for role in self.predict_dir.keys():
            if not os.path.isdir(self.predict_dir[role]):
                os.makedirs(self.predict_dir[role])

        self.print_log_line("Start", 0, "Start", 0)

    def close(self):
        """
        closes log file
        """
        self.print_and_log_simple("closing log file %s" % self.log_file_name)
        self.log_file.close()
        if self.logs_list_name is not None:
            with open(self.logs_list_name,'a+') as fid:
                fid.write(self.log_file_name + '\n')

    def initialize_log_file(self):
        """
        Finds the next unused number for log file and initializes a log file by that name
        (e.g. 241.log). Copies the current version of this file into the log file.
        """

        self.machine_ind = 0  # default value

        # range of run_numbers (used as log file names, etc.) 
		range = [0, 1000000]

        # enum for use with range
        START = 0
        END = 1

        raw_existing_log_file_numbers = \
            [int(nm.replace('.log', '').replace(CWN_env.logs_dir, '')) \
             for nm in glob.glob(CWN_env.logs_dir + '*.log')]

        existing_log_file_numbers = [num for num in raw_existing_log_file_numbers if
                                     ((range[START] <= num) and (num < range[END]))]

        if len(existing_log_file_numbers) == 0:
            self.run_number = range[START]
        else:
            self.run_number = 1 + max(existing_log_file_numbers)
        self.log_file_name = CWN_env.logs_dir + str(self.run_number) + '.log'

        CodeSnapshots.snapshot_this_file(self.run_number)

        self.log_file = open(self.log_file_name, 'a')
        print("opening log file %s" % self.log_file_name)

        with open(os.path.realpath(__file__)) as fid:
            copyfileobj(fid, self.log_file)

    # =================================================================
    #                          TRAIN
    # =================================================================

    def build_feed_dict(self, role, from_, to_, iter_ind=0, is_training=True, index_list=None):
        """
        create minibatch of samples

        :param role: either "train" or "validation" (which data to use)
        :param from_: first index in sample array
        :param to_: one-after-last index in sample array
        :param is_training: if true, run network in train mode (affects Batchnorm)
        :param index_list: if supplied, then from_ and to_ define a range in this list of indexes
        :return: feed_dict to be used with session.run()
        """

        feed_dict = {}
        for fieldname in ['images', 'noisy_labels', 'true_labels', 'is_inlier']:
            key = self.graph.get_tensor_by_name(fieldname + ':0')
            if index_list is not None:
                val = self.D.data[role][fieldname][index_list[from_:to_], ...]
            else:
                val = self.D.data[role][fieldname][from_:to_, ...]
            feed_dict[key] = val

        feed_dict[self.batchnorm_training] = is_training

        # calculate learning rate:
        current_epoch = (1.0 * iter_ind) / self.iters_per_epoch
        decrease_times = np.floor(current_epoch / self.P['solver']['epochs_till_decrease'])
        decrease_times = min(decrease_times, self.P['solver']['max_times_to_decrease'])
        total_decrease_ratio = ((1.0 / self.P['solver']['decrease_factor']) ** decrease_times)
        learning_rate = self.P['solver']['learning_rate'] * total_decrease_ratio
        feed_dict[self.learning_rate] = learning_rate
        return feed_dict


    def order_train_data(self, order):
        for key in self.D.data['train'].keys():
            try:
                self.D.data['train'][key] = self.D.data['train'][key][order, ...]
            except TypeError as E:
                if "'int' object has no attribute '__getitem__'" not in E:
                    raise E

    def train(self):
        """
        Run training iterations.
        Behavior is controlled by "params" dictionary (see top of file)
        """

        self.history = {'validation_acc': [], 'iters': []}
        self.early_stop = {'highest_acc':0.0, 'now': False}

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(graph=self.graph,config = config) as sess:

            # initialize variables, either from checkpoint or randomly:
            if self.P['saver']['restore']:
                self.print_and_log('restoring from latest checkpoint')
                self.saver.restore(sess,
                                   tf.train.latest_checkpoint(CWN_env.checkpoints_dir))
            else:
                sess.run(tf.global_variables_initializer())

            loss_tensors = tf.get_collection(tf.GraphKeys.LOSSES) \
                           + tf.get_collection('DEBUG_PRINTS')

            train_tensors = (
                loss_tensors
                + tf.get_collection(tf.GraphKeys.TRAIN_OP)
            )

            assert len(tf.get_collection(tf.GraphKeys.TRAIN_OP))==1, 'There should be exactly one train op. Otherwise, revise loss_vals.pop() line in the code'

            TrainingDone = False

            to_ = 0
            for i in xrange(self.num_iters):

                # get next minibatch:
                from_ = to_
                to_ = from_ + self.P['net']['batch_size']
                if from_ == self.D.data['train']['num_samples']:
                    # shuffle training data and return to start
                    order = np.random.permutation(self.D.data['train']['num_samples'])
                    self.order_train_data(order)
                    self.ord = self.ord[order]

                    from_ = 0
                    to_ = self.P['net']['batch_size']
                elif to_ > self.D.data['train']['num_samples']:
                    # last batch is smaller then other batches
                    to_ = self.D.data['train']['num_samples']

                feed_dict = self.build_feed_dict('train', from_, to_, i, is_training=True)

                # perform training
                loss_vals = sess.run(train_tensors, feed_dict=feed_dict)
                loss_vals.pop()  # remove unnecessary train_op output

                # report accuracies:
                special_report_time =  ((i == (self.num_iters - 1))
                            or (i in self.P['predict']['general']['when']))

                if special_report_time or (np.mod(i, self.P['report']['train']) == 0):
                    self.report_train(sess, feed_dict, i, loss_tensors, loss_vals)
                if special_report_time or (np.mod(i, self.P['report']['validation']) == 0):
                    self.report_accuracy_on_whole_set('validation',sess, i)
                    self.report_accuracy_on_whole_set('test',sess, i)
                if (np.mod(i, self.P['report']['whole_train_set']) == 0):
                    self.report_accuracy_on_whole_set('train', sess, i)

                # predict (on pre-defined iterations):
                if i in self.P['predict']['general']['when']:
                    for role in ['train','validation','test']:
                        if self.P['predict']['general']['output_features']:
                            self.output_feature(sess, i, role, str(i))
                        if self.P['predict']['general']['output_predictions']:
                            self.output_prediction(sess, i, role, str(i))

                # predict (on early-stop iteration (a.k.a "best")):
                if self.early_stop['now']:
                    self.early_stop['now'] = False
                    for role in ['train', 'test', 'validation']:
                        with open(self.predict_dir[role] + 'best_iteration.txt', 'w') as fid:
                            fid.write(str(i) + '\n')
                        if self.P['predict']['best']['output_features']:
                            self.output_feature(sess, i, role, 'best')
                        if self.P['predict']['best']['output_weights']:
                            self.output_weights(sess, i, role, 'best')
                        if self.P['predict']['best']['output_predictions']:
                            self.output_prediction(sess, i, role, 'best')

                # save checkpoint
                if ((np.mod(i, self.P['saver']['interval']) == 0) and (i > 0)) or (i == (self.num_iters - 1)):
                    checkpoint_filename = CWN_env.checkpoints_dir + 'checkpoint_' + datetime.datetime.now().strftime("%H_%M_%S")
                    self.saver.save(sess, checkpoint_filename, global_step=i)
                    self.print_and_log("Saving checkpoint", i)

                if TrainingDone:
                    break

    def report_train(self, sess, feed_dict, i, loss_tensors, loss_vals):

        not_train_tensors = (
            loss_tensors
            + [self.learning_rate])

        # when using BatchNorm the behaviour is different when
        # not training. Run the net again on this mini-batch in not-training
        # mode to get the NotTraining losses and predictions
        feed_dict[self.batchnorm_training] = False
        loss_vals_not_training = sess.run(not_train_tensors, feed_dict=feed_dict)
        learning_rate = loss_vals_not_training.pop()

        vals = {}
        vals['Train'] = loss_vals
        vals['NotTrain'] = loss_vals_not_training

        # print out losses for this minibatch
        self.print_and_log("learning_rate = %f" % learning_rate, i)
        for typ in ['Train', 'NotTrain']:
            for t, v in zip(loss_tensors, vals[typ]):
                self.print_log_line(typ, i, self.meaningful_names.get(t.name, t.name), v)


    # =================================================================
    #                          PREDICT
    # =================================================================

    def output_weights(self, sess, train_iter, mode, when):
        feature_tensors = tf.get_collection('PREDICTION_WEIGHTS')
        features = sess.run(feature_tensors)
        all_features = {t:f for t,f in zip(feature_tensors,features)}

        for t in feature_tensors:
            name = self.meaningful_names[t.name]
            feature_filename = self.predict_dir[mode] + name + '.' + when + '.' + mode + ".bin"
            with open(feature_filename, 'wb') as fid:
                all_features[t].tofile(fid)
            self.print_and_log("wrote to file " + feature_filename, train_iter)
            format_filename = feature_filename.replace('.bin','.format.txt')
            with open(format_filename, 'w') as fid:
                fid.write("%s %s\n" % (all_features[t].dtype, all_features[t].shape))

    def output_feature(self, sess, train_iter, mode, when):
        """
        :param sess: 
        :return: 
        """
        assert mode in ['train', 'validation',
                        'test'], "argument 'mode' for predict() must be either 'train' or 'validation' or 'test'"
        data_name = mode

        if self.D.data[data_name] is None:
            return

        if mode == 'train':
            #re-order train data in original order (before shuffling)
            inv_ord = np.argsort(np.arange(len(self.ord))[self.ord])
            self.order_train_data(inv_ord)

        feature_tensors = tf.get_collection('PREDICTION_LAYERS')
        # define number of samples:
        num_samples = self.D.data[data_name]['num_samples']

        index_list = np.arange(num_samples)

        self.print_and_log("predicting features ('%s' mode)..." % mode, train_iter)
        to_ = 0

        batch_size = self.P['net']['batch_size']
        all_features = {}
        feature_shape = {}
        for t in feature_tensors:
            feature_shape[t] = t.get_shape().as_list()
            all_features_shape = feature_shape[t]
            all_features_shape[0] = num_samples
            all_features[t] = np.zeros(all_features_shape, dtype=np.float32)

        while to_ < num_samples:
            from_ = to_
            to_ = from_ + batch_size
            if to_ > num_samples:
                to_ = num_samples
            cur_batch_size = to_-from_

            # get minibatch
            feed_dict = self.build_feed_dict(
                data_name,
                from_,
                to_,
                is_training=False,
                index_list=index_list)

            features = sess.run(feature_tensors, feed_dict=feed_dict)
            for t,v in zip(feature_tensors, features):
                shape = feature_shape[t]
                shape[0] = cur_batch_size
                all_features[t][from_:to_,...] = np.reshape(v, shape)
            if np.mod(to_,10000) ==0:
                self.print_and_log("so far %d samples out of %d" % (to_, num_samples), train_iter)

        for t in feature_tensors:
            name = self.meaningful_names[t.name]
            feature_filename = self.predict_dir[mode] + name + '.' + when + '.' + mode + ".bin"
            with open(feature_filename, 'wb') as fid:
                all_features[t].tofile(fid)
            self.print_and_log("wrote to file " + feature_filename, train_iter)
            format_filename = feature_filename.replace('.bin','.format.txt')
            with open(format_filename, 'w') as fid:
                fid.write("%s %s\n" % (all_features[t].dtype, all_features[t].shape))

        if mode == 'train':
            self.order_train_data(self.ord)

    def output_data_for_probs(self, role='test'):
        """
        Outputs:
        1. Confusion Matrix
        2. Ground truth labels for test set
        
        :return: 
        """
        try:
            ConfusionMatrix = self.D.corruption_matrix

            lbls = self.D.data[role]['true_labels']
            LABELS_FILENAME = self.predict_dir[role] + 'Labels1Hot.' + str(lbls.shape[0]) + "." + str(lbls.shape[1]) + ".bin"
            with open(LABELS_FILENAME, 'w') as fid:
                self.D.data[role]['true_labels'].astype('bool').tofile(fid)

            CONF_MAT_FILENAME = self.predict_dir[role] + 'ConfusionMatrix.' + str(ConfusionMatrix.shape[0]) + "." + str(lbls.shape[1]) + ".bin"
            with open(CONF_MAT_FILENAME, 'w') as fid:
                ConfusionMatrix.astype('float32').tofile(fid)
        except AttributeError as E:
            pass

    def output_prediction(self, sess, train_iter, mode, when):
        """

        :param sess: 
        :return: 
        """
        assert mode in ['train', 'validation','test'], "argument 'mode' for predict() must be either 'train' or 'validation' or 'test'"
        data_name = mode

        if self.D.data[data_name] is None:
            return

        if mode == 'train':
            inv_ord = np.argsort(np.arange(len(self.ord))[self.ord])
            self.order_train_data(inv_ord)

        # define number of samples:
        num_samples = self.D.data[data_name]['num_samples']

        index_list = np.arange(num_samples)

        self.print_and_log("predicting ('%s' mode)..." % mode, train_iter)

        batch_size = self.P['net']['batch_size']
        prediction_filename = self.predict_dir[mode] + 'Prediction.' + when + '.' + mode + ".bin"
        prediction_file = open(prediction_filename, 'wb')

        all_predictions = np.zeros([num_samples, self.D.num_classes], dtype=np.float32)
        to_ = 0
        while to_ < num_samples:
            from_ = to_
            to_ = from_ + batch_size
            if to_ > num_samples:
                to_ = num_samples
            cur_batch_size = to_-from_

            # get minibatch
            feed_dict = self.build_feed_dict(
                data_name,
                from_,
                to_,
                is_training=False,
                index_list=index_list)

            # predict in NotTrain mode
            predictions = sess.run(self.predict, feed_dict=feed_dict)
            all_predictions[from_:to_, :] = np.reshape(predictions, [cur_batch_size, self.D.num_classes])
            if np.mod(from_,10000) ==0:
                self.print_and_log(str(from_))

        all_predictions.tofile(prediction_file)

        self.print_and_log("wrote to file " + prediction_file.name)

        prediction_file.close()

        if mode == 'train':
            self.order_train_data(self.ord)

    # =================================================================
    #                          VALIDATION
    # =================================================================
    def report_accuracy_on_whole_set(self, role, sess, iter):
        """
        Calculate loss/accuracy on entire set 

        :param sess: tensorflow session (contains current state of net)
        :param iter: current training iteration number (for printouts)
        """
        aggregate_loss = 0.0
        to_ = 0
        num_samples = self.D.data[role]['num_samples']
        if num_samples == 0:
            return
        batch_size = self.P['net']['batch_size']
        loss_tensors_simple = tf.get_collection(tf.GraphKeys.LOSSES)
        loss_tensors_corrupt = tf.get_collection("LOSSES_ON_CORRUPT_SAMPLES")
        loss_tensors = loss_tensors_simple + loss_tensors_corrupt
        num_simple_losses = len(tf.get_collection(tf.GraphKeys.LOSSES))

        aggregate_count = {
            'num_correct_corrupt': 0,
            'num_total_corrupt': 0,
            'num_correct_unchanged': 0,
            'num_total_unchanged': 0
        }
        
        while to_ < num_samples:
            from_ = to_
            to_ = from_ + batch_size
            if to_ > num_samples:
                to_ = num_samples

            feed_dict = self.build_feed_dict(role, from_, to_, is_training=False)

            loss_vals = sess.run(loss_tensors, feed_dict=feed_dict)
            loss_vals_simple = loss_vals[:num_simple_losses]
            loss_vals_corrupt = loss_vals[num_simple_losses:]

            aggregate_loss += (to_ - from_) * np.array(loss_vals_simple)

            for t,v in zip(loss_tensors_corrupt, loss_vals_corrupt):
                nm = self.meaningful_names.get(t.name, t.name)
                aggregate_count[nm] += v

        if role == "train":
            role = "WholeTrain"
        mean_loss = aggregate_loss / num_samples
        for t, mean_v in zip(loss_tensors_simple, mean_loss):
            loss_name = self.meaningful_names[t.name]
            self.print_log_line(role.capitalize(), iter, loss_name, mean_v)
            
        if aggregate_count['num_total_corrupt'] > 0:
            corrupt_accuracy = aggregate_count['num_correct_corrupt']/aggregate_count['num_total_corrupt']
            unchanged_accuracy = aggregate_count['num_correct_unchanged']/aggregate_count['num_total_unchanged']
            self.print_log_line(role.capitalize(), iter, "accuracy_on_corrupt_samples", np.around(corrupt_accuracy,4))
            self.print_log_line(role.capitalize(), iter, "accuracy_on_unchanged_samples", np.around(unchanged_accuracy,4))
            
        if role == 'validation':
            for t, v in zip(loss_tensors_simple, mean_loss):
                if (self.meaningful_names.get(t.name, t.name) == 'true_accuracy'):
                    self.history['validation_acc'].append(v)
                    self.history['iters'].append(iter)

            i = iter
            iters = np.array(self.history['iters'])
            acc_vals = np.array(self.history['validation_acc'])
            if (i >= self.P['early_stop']['initial_iters_to_skip']):
                # filtering: calc median of last 'iters_to_smooth' iters
                is_in_filter_window = (iters > i- self.P['early_stop']['iters_to_smooth'])
                is_in_filter_window[-self.P['early_stop']['min_vals_to_smooth']:] = True
                newest_acc = np.median(acc_vals[is_in_filter_window])
                if (self.early_stop['highest_acc'] < newest_acc):
                    self.early_stop['highest_acc'] = newest_acc
                    self.early_stop['now'] = True

    # =================================================================
    #                          PRINT-OUTS
    # =================================================================
    def print_log_line(self, role_str, iter, loss_name, loss_val):
        """
        Prints line to log/screen
        :param role_str: either 'Train' or 'Validation'
        :param iter: current iteration number
        :param loss_name: e.g. 'b_loss'
        :param loss_val: e.g. 0.00213
        """
        s = "%s: %s: " % (role_str, loss_name) + ' ' + str(loss_val)
        self.print_and_log(s, iter)

    def print_and_log(self, str, iter=-1):
        prefix = "Time: %s Iter: %d " % (
            datetime.datetime.now().strftime("%H:%M:%S.%f"),
            iter)
        str = prefix + str
        self.print_and_log_simple(str)

    def print_and_log_simple(self, str):
        print(str)
        self.log_file.write(str + '\n')
        self.log_file.flush()

    # =================================================================
    #                          MAIN
    # =================================================================
    def go(self, noise_level=0.0):
        """
        Main function. Loads data, builds net, and trains
        """
        self.noise_level = noise_level

        self.load_data()
        self.ord = np.arange(self.D.data['train']['num_samples']) # to keep track of shuffling
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.build_net()

        self.train()
        self.close()
        return self.run_number

if __name__ == "__main__":

    if isinstance(P['data']['noise_type'],list):
        noise_type = P['data']['noise_type'][0]
    else:
        noise_type = P['data']['noise_type']

    for repeat_num in xrange(P['experiment']['repeats']):
        for noise_level in NOISE_LEVELS[noise_type]:
            engine = CnnWithNoise(P, repeat_num)
            engine.go(noise_level)
