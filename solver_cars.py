from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import mmd
import msda
from torch.autograd import Variable
from model.build_gen import *
from datasets.cars import cars_combined
import numpy as np
import math


class Solver(object):
    def __init__(self, args, batch_size=64,
                 target='mnist', learning_rate=0.0002, interval=100, optimizer='adam'
                 , checkpoint_dir=None, save_epoch=10):
        self.batch_size = batch_size
        self.target = target
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff

        print('dataset loading')
        if args.data=='digits':
            if args.dl_type == 'original':
                self.datasets, self.dataset_test = dataset_read(target, self.batch_size)
            elif args.dl_type == 'hard_cluster':
                self.datasets, self.dataset_test = dataset_hard_cluster(target, self.batch_size)
            elif args.dl_type=='soft_cluster':
                self.datasets, self.dataset_test = dataset_combined(target, self.batch_size)
            else:
                raise Exception('Type of experiment undefined')

        elif args.data=='cars':
            if args.dl_type == 'soft_cluster':
                self.datasets, self.dataset_test = cars_combined(target,self.batch_size)