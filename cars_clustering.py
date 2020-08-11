from __future__ import print_function
from datasets.cars import cars_combined

datasets, dataset_test, dataset_valid, classwise_dataset = cars_combined('CCSurv',128,"/data/CompCarsCropped/data_cropped",0,0)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import os
import sys

lengthForPCA = 600
small_dimension = 30
arrayOfClusterstorch = []
print('starting to read dataloader')
for batch_idx, data in enumerate(datasets):
	img_s = data['S'].cuda()
	arrayOfClusterstorch.append(img_s)
	if(len(arrayOfClusterstorch) > lengthForPCA):
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

datasets, dataset_test, dataset_valid, classwise_dataset = cars_combined('CCSurv',128,"/data/CompCarsCropped/data_cropped",0,0)
arrayOfClusterstorch = []
arrayOfPaths = []
print('starting to read dataloader again')
for batch_idx, data in enumerate(datasets):
	img_s = data['S'].cuda()
	arrayOfClusterstorch.append(img_s)
	arrayOfPaths += data['S_paths']

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
print(a_dictionary)





















