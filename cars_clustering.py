from __future__ import print_function
from datasets.cars import cars_combined
datasets, dataset_test, dataset_valid, classwise_dataset = cars_combined('CCSurv',128,"/data/CompCarsCropped/data_cropped",0,10)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import os

for batch_idx, data in enumerate(datasets):
	img_s = Variable(data['S'])
	img_path = Variable(data['S_paths'])
	print(img_s)
	print(img_path)
	print(img_s.size())


