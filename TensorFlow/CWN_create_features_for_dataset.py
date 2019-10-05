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

# Some noise types require that each sample in a dataset have a
# feature vector associated with it, e.g. to allow measuring distances
# between samples. This function trains a network on the dataset, and then stores
# the output of the one-before-last layer into a features file in the directory
# of the dataset.
# In addition, it stores the network's prediction for each sample, which can also
# be useful in defining some noise types.

import numpy as np
from CnnWithNoise import CnnWithNoise
from CnnWithNoise import P as default_params
import CWN_env
from copy import deepcopy
import os
from shutil import copyfile

SHARED_EXPERIMENT_PARAMS = {
    'early_stop': {     'min_vals_to_smooth': 1,
                        'iters_to_smooth': 1,
                        'initial_iters_to_skip': 1000},
    'predict': {        'features_after_relu': False,
                        'best': {   'output_predictions': True,
                                    'output_weights': False,
                                    'output_features': True},
                        'output_data_for_probs': False,
                        'general': {'when': [],
                                    'output_predictions': False,
                                    'output_features': False}},
    'experiment': { 'repeats': 1},
    'report': {     'train': 10,
                    'whole_train_set': np.nan,
                    'validation': 10},
    'net': {        'batchnorm_momentum': 0.99,
                    'batch_size': 256},
    'data': { 'noise_type': 'none'}
}

PARAMS = {}
PARAMS['mnist'] = {
    'special': {'feature_file_prefix': 'fc1'},
    'data': {   'dataset': 'mnist',
                'normalization': '0.0_to_1.0' },
    'net':  {   'main_structure': 'lenet'   },
    'train':        {            'epochs': 20,        },
    'solver':   {   'learning_rate': 0.01,
                    'epochs_till_decrease':10,
                    'decrease_factor': 2,
                    'max_times_to_decrease': np.inf,
                    'l2_reg': 10 ** -7}}

PARAMS['cifar10'] = {
    'special': {'feature_file_prefix': 'feature'},
    'data': {   'dataset': 'cifar10',
                'normalization': 'per_pixel_mean_std',
                'noise_type': 'none'},
    'net':  {   'main_structure': 'allconv'},
    'train':{   'epochs': 60 },
    'solver':   {   'learning_rate': 0.007,
                    'epochs_till_decrease': 30,
                    'decrease_factor': 2,
                    'max_times_to_decrease': np.inf,
                    'l2_reg': 10 ** -7 }}

def combine_params(base, override):
    res = deepcopy(base)
    for k1 in override.keys():
        if isinstance(override[k1],dict):
            if k1 not in res.keys():
                res[k1] = {}
            for k2 in override[k1].keys():
                if isinstance(override[k1][k2], dict):
                    if k2 not in res[k1].keys():
                        res[k1][k2] = {}
                    for k3 in override[k1][k2].keys():
                        res[k1][k2][k3]=override[k1][k2][k3]
                else:
                    res[k1][k2] = override[k1][k2]
        else:
            res[k1] = override[k1]
    return res

def go():
    for dataset_name in ['mnist', 'cifar10']:
        P0 = combine_params(SHARED_EXPERIMENT_PARAMS, PARAMS[dataset_name])
        P = combine_params(default_params, P0)
        engine = CnnWithNoise(P)
        run_number = engine.go()
        # TODO - copy predict and feature files to correct location

        for role in ['train','validation','test']:
            pred_dir = CWN_env.predictions_dir + ('%s/%d/' % (role,run_number))
            predictions_file = pred_dir + "Prediction.best.%s.bin" % role
            if os.path.isfile(predictions_file):
                output_filename = CWN_env.data_dir + "%s/%s.predictions.%s.bin" % (dataset_name, dataset_name, role)
                copyfile(predictions_file, output_filename)
            else:
                print "can't find predictions file: %s" % predictions_file
            features_file = pred_dir + P['special']['feature_file_prefix'] + (".best.%s.bin" % role)
            if os.path.isfile(features_file):
                output_filename = CWN_env.data_dir + "%s/%s.features.%s.bin" % (dataset_name, dataset_name, role)
                copyfile(features_file, output_filename)
            else:
                print "can't find features file: %s" % features_file

if __name__ == "__main__":
    go()