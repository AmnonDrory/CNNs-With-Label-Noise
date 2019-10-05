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

from CWN_env import data_dir
from keras.datasets import mnist as keras_mnist
from CWN_dataset import CWN_dataset

class CWN_mnist(CWN_dataset):
    def __init__(self):
        self.dataset_name = 'mnist'
        self.keras_loader = keras_mnist
        self.wid = 28
        self.hei = 28
        self.depth = 1
        self.data_dir = data_dir + 'mnist/'
        self.label_bytes = 1
        self.num_classes = 10
        self.validation_split = 0.1
        self.maybe_download()
        self.train_labels_file = None
        self.load_from_files()

if __name__ == '__main__':
    E = CWN_mnist()