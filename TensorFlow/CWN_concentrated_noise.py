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
from sklearn.cluster import KMeans

def largest(s):
    return np.array(s.values()).max()

def evaluate_subclass_size(is_subclass, cur_is_test, cur_is_validation, num_samples, num_classes, noise_level, min_size_diff):

    sub_is = {}
    sub_is['test'] = cur_is_test[is_subclass]
    sub_is['validation'] = cur_is_validation[is_subclass]
    sub_is['train'] = ~( sub_is['test'] | sub_is['validation'] )

    size_diff = { k: None for k in sub_is.keys() }

    for key in sub_is.keys():
        expected_num = num_samples[key] * (1.0 / num_classes) * noise_level
        actual_num = sub_is[key].sum()
        size_diff[key] = np.abs((actual_num/expected_num)-1)

    if min_size_diff is None:
        is_better = True
    else:
        is_better = largest(size_diff) < largest(min_size_diff)

    if is_better:
        min_size_diff = size_diff

    return min_size_diff, is_better

def generate_concentrated_noise(noise_level, feature, GT_labels, num_classes):
    """

    :param noise_level: a fraction. such as 0.12
    :param features: dict, in each cell: NxD array, where N is the number of samples, and D the length of the feature vector for each sample (e.g. 256)
    :param GT_labels_1hot: dict, in each cell: NxC array, where C is the number of possible classes, and each row encodes the class for the corresponding sample in 1-hot encoding
    :return:
    """

    roles = ['train', 'validation', 'test']
    num_subclasses = int(np.round(1.0/noise_level)) # we perform k-means to arrive at approximately the required noise level
    if num_subclasses < 2:
        num_subclasses = 2

    num_samples = {}
    is_validation = {}
    is_test = {}
    for role in roles:
        num_samples[role] = len(GT_labels[role])
        is_validation[role] = np.zeros([num_samples[role],1], dtype=np.bool) + (role=='validation')
        is_test[role] = np.zeros([num_samples[role],1], dtype=np.bool) + (role == 'test')

    from_ = [0, num_samples['train']]
    from_.append( from_[1] + num_samples['validation'] )
    to_ = from_[1:]
    to_.append( to_[1] +  num_samples['test'] )
    A_num_samples = to_[2]
    range = {key: slice(from_[i],to_[i]) for i, key in enumerate(['train','validation','test'])}

    A_feature = np.vstack([feature[role] for role in roles])
    A_labels = np.hstack([GT_labels[role] for role in roles])
    A_is_validation = np.vstack([is_validation[role] for role in roles])
    A_is_test = np.vstack([is_test[role] for role in roles])

    A_noisy_labels = A_labels.copy()
    A_is_inlier = np.ones(A_num_samples, dtype=np.bool)

    MAX_LOOPS = 10
    TOLERANCE = 0.1

    for original_class in xrange(num_classes):
        alternative_class = np.mod(original_class + num_classes / 2, num_classes)

        is_orig_class = (A_labels == original_class)
        orig_class_inds = is_orig_class.nonzero()[0]
        
        cur_features = A_feature[is_orig_class, :]
        cur_is_test = A_is_test[is_orig_class]
        cur_is_validation = A_is_validation[is_orig_class]

        min_size_diff = None
        reached_tolerance = False
        for loop in xrange(MAX_LOOPS):  # repeat until a subset of the expected size is found
            engine = KMeans(n_clusters=num_subclasses, init='k-means++', n_init=20, precompute_distances=True,
                            verbose=0, n_jobs=-1)
            kmeans_res = engine.fit(cur_features)
            kmeans_labels = kmeans_res.labels_

            for left_handed_subclass in xrange(num_subclasses):

                is_subclass =  (kmeans_labels == left_handed_subclass)
                left_handed_inds = orig_class_inds[is_subclass]

                min_size_diff, is_better = evaluate_subclass_size(is_subclass, cur_is_test, cur_is_validation, num_samples, num_classes, noise_level, min_size_diff)

                if is_better:
                    left_handed_inds_with_min_size_diff = left_handed_inds

                if largest(min_size_diff) < TOLERANCE :
                    reached_tolerance = True
                    break

            if reached_tolerance:
                break

        left_handed_inds = left_handed_inds_with_min_size_diff

        A_noisy_labels[left_handed_inds] = alternative_class
        A_is_inlier[left_handed_inds] = False

    # now separate results to train, validation, etc.
    
    res = {}
    for role in roles:
        res[role] = {}
        res[role]['noisy_labels'] = A_noisy_labels[range[role]]
        res[role]['is_inlier'] = A_is_inlier[range[role]]

    return res