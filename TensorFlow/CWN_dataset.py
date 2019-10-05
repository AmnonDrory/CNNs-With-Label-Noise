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
from glob import glob
from copy import deepcopy

class CWN_dataset:
    def load_from_files(self):
        """
        :return: 
        """
        data_types = ['labels','images','predictions','features']
        data_class = {'labels': np.uint8, 'images': np.uint8, 'predictions': np.float32, 'features':np.float32}
        roles = ['train', 'test', 'validation']
        D = {k:{} for k in roles}
        N = {k: None for k in roles}
        FN = deepcopy(D)
        for data_type in data_types:
            for role in D.keys():
                fn_ = glob(self.data_dir + "%s.%s.%s*bin" % (self.dataset_name, data_type, role))
                if len(fn_) == 0:
                    continue
                else:
                    fn = fn_[0]
                FN[role][data_type] = fn
                if data_type in ['labels','images']:
                    N[role] = int(fn.split('.')[3])
                with open(fn, 'rb') as fid:
                    raw_data = np.fromfile(fid, data_class[data_type])
                if data_type == 'labels':
                    assert raw_data.size == N[role]
                    D[role][data_type] = raw_data
                elif data_type == 'images':
                    D[role][data_type] = np.reshape(raw_data, [N[role], self.hei, self.wid, self.depth])
                else:
                    D[role][data_type] = np.reshape(raw_data, [N[role], -1])

        if self.train_labels_file is not None:
            with open(self.train_labels_file, "r") as fid:
                raw = np.fromfile(fid, np.uint8)
            labels = np.reshape(raw, D['train']['labels'].shape)
            D['train']['original_labels'] = D['train']['labels'].copy()
            D['train']['labels'] = labels

        self.D = D
        self.N = N
        self.FN = FN

    def pseudo_random_order(self,ims):
        # reorder samples in a psudo-random way, but consistent across runs
        image_sums = np.array([np.sum(ims[i, ...]) for i in xrange(ims.shape[0])],
                              dtype=np.float32)
        image_code = np.sin(image_sums)
        ord0 = np.argsort(image_code)
        ord = np.argsort(np.sin(ord0.astype(float)))
        return ord

    def split_train_validation(self, images, labels):
        ims = images['train']
        lbls = labels['train']
        ord = self.pseudo_random_order(ims)
        ims = ims[ord,...]
        lbls = lbls[ord,...]
        is_validation = np.zeros([ims.shape[0]], np.bool)

        for cls in xrange(self.num_classes):
            is_class = (lbls == cls)
            N_cls = np.sum(is_class)
            N_validation = int(np.round(self.validation_split * N_cls))
            np.where(is_class)[0][:N_validation]
            is_validation[np.where(is_class)[0][:N_validation]]=True

        images['validation'] = ims[is_validation,...]
        labels['validation'] = lbls[is_validation,...]
        images['train'] = ims[~is_validation,...]
        labels['train'] = lbls[~is_validation,...]
        print "total validation samples: %d" % images['validation'].shape[0]
        print "total train samples: %d" % images['train'].shape[0]
        return images, labels

    def maybe_download(self):
        dest_directory = self.data_dir
        data_exists = True
        if os.path.exists(dest_directory):
            files = glob(self.data_dir + '*.bin')
            all_files = '|'.join(files)
            for data in ['images', 'labels']:
                for role in ['train', 'validation', 'test']:
                    if not (self.dataset_name + '.%s.%s.' % (data, role)) in all_files:
                        data_exists = False
        else:
            data_exists = False
            os.makedirs(dest_directory)

        if data_exists:
            return
        else:
            print "Using Keras to get dataset " + self.dataset_name

        images = {}
        labels = {}
        (images['train'], labels['train']), (images['test'], labels['test']) = self.keras_loader.load_data()

        images, labels = self.split_train_validation(images, labels)

        for role in images.keys():
            labels_fn = dest_directory + self.dataset_name + '.labels.%s.%s.bin' % (role, labels[role].shape[0])
            with open(labels_fn, 'wb') as fid:
                labels[role].astype('uint8').tofile(fid)

            images_fn = labels_fn.replace('labels','images')
            with open(images_fn, 'wb') as fid:
                images[role].astype('uint8').tofile(fid)