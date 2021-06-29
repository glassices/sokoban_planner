from common import config
from network import Network
import utils

import os
import copy
import random
import argparse
import numpy as np
import hashlib
import subprocess
import datetime
import glob
import logging
from utils import AverageMeter, ProgressMeter

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.cpp_extension

parser = argparse.ArgumentParser(description='Used for resuming training')
parser.add_argument('--resume', default=None, type=int, metavar='ITER',
                    help='done iteration to resume from')
args = parser.parse_args()

# append md5 of cpp file can automatically recompile JIT every
# time we modify cpp file

heads = glob.glob('cpp/*.hpp')
sources = glob.glob('cpp/*.cpp')
hash_code = hashlib.md5(b''.join([open(head, 'rb').read() for head in heads] +
                                 [open(source, 'rb').read() for source in sources])).hexdigest()
sokoban_cpp = torch.utils.cpp_extension.load(name='sokoban_cpp_{}'.format(hash_code),
                                             sources=sources,
                                             extra_cflags=['-std=c++17', '-O6'])
def main():
    pass

if __name__ == '__main__':
    main()
