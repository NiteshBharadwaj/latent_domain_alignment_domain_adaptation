from __future__ import print_function
from datasets.cars import cars_combined_real

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
print('MAKING dataloader')
datasets, dataset_test, dataset_valid, classwise_dataset = cars_combined_real('CCSurv',128,"/data/CompCarsCropped/data_cropped",0,args.num_workers)

lengthForPCA = 1000
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
sys.exit()

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


print('MAKING dataloader')
datasets, dataset_test, dataset_valid, classwise_dataset = cars_combined('CCSurv',128,"/data/CompCarsCropped/data_cropped",0,args.num_workers)
arrayOfClusterstorch = []
arrayOfPaths = []
print('starting to read dataloader again')
totalTillNow = 0
shortened_numpys = []
for batch_idx, data in enumerate(datasets):
	img_s = data['S'].cuda()
	arrayOfClusterstorch.append(img_s)
	arrayOfPaths += data['S_paths']
	totalTillNow += img_s.size()[0]
	print(img_s.size())
	if(totalTillNow > 1000):
		print(totalTillNow)
		concatenatedTensor = torch.cat(tuple(arrayOfClusterstorch), 0)
		concatenatedTensor = torch.reshape(concatenatedTensor, (concatenatedTensor.size()[0],-1))
		X_vec = concatenatedTensor.cpu().data.numpy()
		X_vec = X_vec - mean
		transformed_data = pca_transformed.transform(X_vec)
		shortened_numpys.append(transformed_data)
		totalTillNow = 0
		arrayOfClusterstorch = []

concatenatedTensor = torch.cat(tuple(arrayOfClusterstorch), 0)
concatenatedTensor = torch.reshape(concatenatedTensor, (concatenatedTensor.size()[0],-1))
X_vec = concatenatedTensor.cpu().data.numpy()
X_vec = X_vec - mean
transformed_data = pca_transformed.transform(X_vec)
shortened_numpys.append(transformed_data)

shortened_numpys = tuple(shortened_numpys)
shortened_numpys = np.concatenate(shortened_numpys, axis=0)

print('starting k means now')
kmeans = KMeans(n_clusters=4, random_state=0).fit(shortened_numpys)

zip_iterator = zip(arrayOfPaths, kmeans.labels_)
a_dictionary = dict(zip_iterator)

print(a_dictionary)
import pickle
with open('/data/ccweb_clusterss.pickle', 'wb') as handle:
    pickle.dump(a_dictionary, handle)





















