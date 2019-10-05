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

import os
import inspect
import time
import shutil
import filecmp
from sys import argv

import CWN_env

__doc__ = """
This file contains a utility for saving a copy of a code file while running it.
This is useful in order to track back the exact version of the code that was run 
for a specific experiment. 
"""

CODE_SNAPSHOT_DIR = CWN_env.code_snapshots_dir
if not os.path.isdir(CODE_SNAPSHOT_DIR):
    os.makedirs(CODE_SNAPSHOT_DIR)

def snapshot_this_file(where=None):
    try:
        assert CODE_SNAPSHOT_DIR[-1] == '/', 'CODE_SNAPSHOT_DIR must end with slash'

        # get the name of the file from which this function was called:
        source_file = inspect.stack()[1][1]
        source_file = os.path.realpath(source_file)

        # determine where to create the snapshot:
        if where is None:
            outdir = CODE_SNAPSHOT_DIR + time.strftime("%d-%m-%y-%H:%M:%S")
        elif type(where) in [int, float]:
            outdir = CODE_SNAPSHOT_DIR + str(int(where))
        elif os.path.isdir(where):
            outdir = where
        else:
            outdir = CODE_SNAPSHOT_DIR + where

        if outdir[-1] != '/':
            outdir += '/'

        # create output directory if necessary
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        target_file = outdir + os.path.split(source_file)[1]

        # if there output directory already contains a file named as the target file,
        # then first rename the existing file:
        backup_file = None
        if os.path.isfile(target_file):
            backup_file = target_file + time.strftime(".%d-%m-%y-%H:%M:%S")
            shutil.copyfile(target_file, backup_file)

        # copy the source file to the output directory
        shutil.copyfile(source_file, target_file)

        if backup_file is not None:
            # no need to keep the backup file if its contents are the same as the new file
            if filecmp.cmp(target_file, backup_file):
                os.remove(backup_file)

        return target_file
    except:
        print "Failed to snapshot code file. Not a deal breaker. Continuing..."