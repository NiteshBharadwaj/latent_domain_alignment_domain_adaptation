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
for batch_idx, data in enumerate(datasets):
	img_s = data['S'].cuda()
	arrayOfClusterstorch.append(img_s)
	if(len(arrayOfClusterstorch) > lengthForPCA):
		break

concatenatedTensor = torch.cat(tuple(arrayOfClusterstorch), 0)
concatenatedTensor = torch.reshape(concatenatedTensor, (concatenatedTensor.size()[0],-1))
X_vec = concatenatedTensor.cpu().data.numpy()


import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


mean = X_vec.mean(0)
X_vec = X_vec - mean
pca_transformed = PCA(n_components=small_dimension).fit(X_vec)

datasets, dataset_test, dataset_valid, classwise_dataset = cars_combined('CCSurv',128,"/data/CompCarsCropped/data_cropped",0,0)
arrayOfClusterstorch = []
arrayOfPaths = []
for batch_idx, data in enumerate(datasets):
	img_s = data['S'].cuda()
	arrayOfClusterstorch.append(img_s)
	arrayOfPaths += data['S_paths']

concatenatedTensor = torch.cat(tuple(arrayOfClusterstorch), 0)
concatenatedTensor = torch.reshape(concatenatedTensor, (concatenatedTensor.size()[0],-1))

X_vec = concatenatedTensor.cpu().data.numpy()
X_vec = X_vec - mean
transformed_data = pca_transformed.transform(complete_data)
kmeans = KMeans(n_clusters=2, random_state=0).fit(transformed_data)
zip_iterator = zip(arrayOfPaths, kmeans.labels_)
a_dictionary = dict(zip_iterator)






















