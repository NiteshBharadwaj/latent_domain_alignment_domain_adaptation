from __future__ import print_function
from datasets.cars import cars_combined

datasets, dataset_test, dataset_valid, classwise_dataset = cars_combined('CCSurv',32,"/data/CompCarsCropped/data_cropped",0,0)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='clustering compcars')
parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                    help='dataloader num_workers')
args = parser.parse_args()

lengthForPCA = 200
small_dimension = 30
arrayOfClusterstorch = []
totalTillNow = 0
print('starting to read dataloader')
for batch_idx, data in enumerate(datasets):
	print('here')
	img_s = data['S'].cuda()
	print(img_s.size())
	arrayOfClusterstorch.append(img_s)
	totalTillNow += img_s.size()[0]
	if(totalTillNow > lengthForPCA):
		break

print('collected for PCA')
concatenatedTensor = torch.cat(tuple(arrayOfClusterstorch), 0)
concatenatedTensor = torch.reshape(concatenatedTensor, (concatenatedTensor.size()[0],-1))
print(concatenatedTensor.size())
X_vec = concatenatedTensor.cpu().data.numpy()
print(X_vec.shape)
print('converted to numpy')

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


mean = X_vec.mean(0)
X_vec = X_vec - mean
pca_transformed = PCA(n_components=small_dimension).fit(X_vec)
print('PCA done')

datasets, dataset_test, dataset_valid, classwise_dataset = cars_combined('CCSurv',128,"/data/CompCarsCropped/data_cropped",0,args.num_workers)
arrayOfClusterstorch = []
arrayOfPaths = []
print('starting to read dataloader again')
totalTillNow = 0
for batch_idx, data in enumerate(datasets):
	img_s = data['S'].cuda()
	arrayOfClusterstorch.append(img_s)
	arrayOfPaths += data['S_paths']
	totalTillNow += img_s.size()[0]
	if(totalTillNow > 200):
		break	

concatenatedTensor = torch.cat(tuple(arrayOfClusterstorch), 0)
concatenatedTensor = torch.reshape(concatenatedTensor, (concatenatedTensor.size()[0],-1))
print('whole data collected')

print(concatenatedTensor.size())


print(len(arrayOfPaths))

X_vec = concatenatedTensor.cpu().data.numpy()
adasdas_vec = X_vec - mean

print('transforming via PCA')
transformed_data = pca_transformed.transform(complete_data)
print('starting k means now')
kmeans = KMeans(n_clusters=2, random_state=0).fit(transformed_data)
zip_iterator = zip(arrayOfPaths, kmeans.labels_)
a_dictionary = dict(zip_iterator)

with open('/data/cars_clusters.pickle', 'wb') as handle:
    pickle.dump(a_dictionary, handle)





















