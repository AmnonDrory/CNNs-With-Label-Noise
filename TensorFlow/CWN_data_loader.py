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
import random
import sys

from CWN_mnist import CWN_mnist
from CWN_cifar10 import CWN_cifar10
from CWN_concentrated_noise import generate_concentrated_noise

class CWN_data_loader():

    def __init__(self, P, noise_level, print_and_log, *args, **kwargs):
        self.P = P
        self.noise_level = noise_level
        self.print_and_log = print_and_log
        self.load_data(*args, **kwargs)

    def load_data(self, normalization=None, train_labels_file=None, noisy_validation=False, *args, **kwargs):
        """
        Loads entire data arrays from files. They are kept in memory throughout
            the life of the program.
        """

        self.data = {}

        if self.P['data']['dataset'] == 'mnist':
            dataset = CWN_mnist()
        elif self.P['data']['dataset'] == 'cifar10':
            dataset = CWN_cifar10(train_labels_file)
        else:
            assert False, "unknown dataset: " + self.P['data']['dataset']

        self.num_classes = dataset.num_classes

        roles = sorted(['train','test','validation'])
        assert (sorted(dataset.D.keys()) == roles)

        self.print_and_log("noise level: " + str(self.noise_level))
        self.print_and_log("randomization: " + str(self.P['data']['noise_type']))

        for role in roles:
            cur_data = dataset.D[role]
            images = cur_data['images']
            labels = cur_data['labels']
            if (self.num_classes > 1):
                labels = labels.flatten()

            self.data[role] = {}
            self.data[role]['true_labels'] = self.labels_to_1hot(labels, self.num_classes)
            if 'original_labels' in cur_data.keys():
                self.data[role]['original_labels'] = self.labels_to_1hot(cur_data['original_labels'], self.num_classes)

            if ((role == 'train') or ((role == 'validation') and noisy_validation)):
                noisy_labels, is_outlier = self.randomize_labels(labels, cur_data)
                self.data[role]['noisy_labels'] = self.labels_to_1hot(noisy_labels, self.num_classes)
                self.data[role]['is_inlier'] = np.logical_not(is_outlier)
            else:
                self.data[role]['noisy_labels'] = self.data[role]['true_labels']
                self.data[role]['is_inlier'] = np.ones([len(labels)],dtype=np.bool)

            self.data[role]['images'] = images

            self.data[role]['num_samples'] = self.data[role]['images'].shape[0]


        if self.P['data']['noise_type'] == 'locally_concentrated':
            self.generate_locally_concentrated_noise(dataset)

        # normalize images:
        self.normalize_images(normalization)

    def normalize_images(self, normalization):

        if normalization == 'per_pixel_mean_std':
            ims = self.data['train']['images'].astype(np.float32)
            per_pixel_mean = np.mean(ims, axis=0, keepdims=True)
            per_pixel_std = np.std(ims, axis=0, keepdims=True)
            epsilon = 10 ** -2
            for role in self.data.keys():
                ims = self.data[role]['images'].astype(np.float32)
                ims = (ims - per_pixel_mean) / (per_pixel_std + epsilon)
                self.data[role]['images'] = ims

        elif normalization == '0.0_to_1.0':
            for role in self.data.keys():
                images = self.data[role]['images']
                brightest = np.iinfo(images.dtype).max
                self.data[role]['images'] = images.astype('float32') / brightest

        else:
            images = self.data['train']['images']
            assert not 'int' in str(images.dtype), "image data is integer but no known method for normalization into float32 is selected"
    
    def create_corruption_matrix(self, main_noise_type, secondary_noise_type):
        L = self.num_classes
        if main_noise_type == 'flip':
            if secondary_noise_type in [None, 'next_index']:
                corruption_matrix = np.eye(L,L,1)
                corruption_matrix[L-1,0] = 1
            elif secondary_noise_type == 'all_to_one':
                corruption_matrix = np.zeros([L,L])
                corruption_matrix[:,0]=1
        elif main_noise_type == 'uniform':
            corruption_matrix = np.full([L,L],1.0/L)
        elif main_noise_type == 'none':
            corruption_matrix = np.eye(L)
        elif main_noise_type == "generalCorruptionMatrix":
            if secondary_noise_type is None: # randomly create a corruption matrix
                corruption_matrix = np.zeros([L, L])
                for i in xrange(L):
                    raw = np.random.rand(L)
                    power = np.random.randint(1,4)
                    emphasized = raw**power
                    prob = emphasized / sum(emphasized)
                    corruption_matrix[i,:] = prob
            else:
                with open(secondary_noise_type,'rb') as fid:
                    raw = np.fromfile(fid, np.float32)
                corruption_matrix = np.reshape(raw, [L, L])
        else:
            corruption_matrix = None # for noise types that are not defined by a corruption matrix
            
        return corruption_matrix

    def randomize_with_corruption_matrix(self, clean_labels, is_noisy, corruption_matrix):
        L = self.num_classes
        res_labels = clean_labels.copy()
        orig_labels = clean_labels[is_noisy]
        new_labels = np.zeros_like(orig_labels)

        for i in xrange(len(orig_labels)):
            orig_lbl = orig_labels[i]
            new_lbl = np.random.choice(np.arange(L),
                                       p=corruption_matrix[orig_lbl, :])
            new_labels[i] = new_lbl

        res_labels[is_noisy] = new_labels
        return res_labels

    def generate_locally_concentrated_noise(self, dataset):
        # prep inputs:
        feature = {}
        true_labels = {}
        for role in dataset.D.keys():
            feature[role] = dataset.D[role]['features']
            true_labels[role] = dataset.D[role]['labels']

        # generate:
        res = generate_concentrated_noise(
            self.noise_level,
            feature,
            true_labels,
            self.num_classes)

        # arrange outputs:
        for role in dataset.D.keys():
            self.data[role]['noisy_labels'] = self.labels_to_1hot(
                                                res[role]['noisy_labels'],
                                                self.num_classes)
            self.data[role]['is_inlier'] = res[role]['is_inlier']


    def randomize_labels(self, labels, cur_data):

        if isinstance(self.P['data']['noise_type'], list):
            main_noise_type = self.P['data']['noise_type'][0]
            secondary_noise_type = self.P['data']['noise_type'][1]
        else:
            main_noise_type = self.P['data']['noise_type']
            secondary_noise_type = None

        self.corruption_matrix = self.create_corruption_matrix(main_noise_type, secondary_noise_type)

        if main_noise_type in ['none', 'locally_concentrated']:
            return labels, np.zeros(shape=[labels.shape[0]],dtype=np.bool)

        if labels.ndim == 2: # in 1-hot-encoding
            clean_labels = np.argmax(labels,axis=1)
        else:
            clean_labels = labels

        N = len(clean_labels)
        is_noisy = np.zeros([N], dtype=np.bool)
        M = int(np.ceil(N * self.noise_level))
        random_inds = random.sample(range(N), M)
        is_noisy[random_inds] = True

        if main_noise_type in ['uniform', 'flip', 'generalCorruptionMatrix']:
            res_labels = self.randomize_with_corruption_matrix(clean_labels, is_noisy, self.corruption_matrix)
        else:
            assert False, 'unknown randomization scheme: ' + str([main_noise_type,secondary_noise_type])

        return res_labels, is_noisy

    def labels_to_1hot(self, labels, num_classes, dtype='uint8'):
        if num_classes <= 1: # regression labels
            return labels
        labels = labels.flatten()
        num_labels = len(labels)
        one_hot = np.zeros([num_labels, num_classes], dtype=dtype)
        one_hot[np.arange(num_labels),labels] = 1
        return one_hot
