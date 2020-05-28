import numpy as np
from scipy.io import loadmat
import gzip
import pickle
import sys
sys.path.append('../utils/')
from utils.utils import dense_to_one_hot
base_dir = './data'
import h5py
def load_usps(directory, use_full=False, usps_less_data_protocol=False):
    base_dir = directory
    if 0:# use_full:
        f = h5py.File('data/usps.h5')

        data_set = [[np.array(f['train']['data']), np.array(f['train']['target'])], 
            [np.array(f['test']['data']), np.array(f['test']['target'])]]
        f.close()
    else: 
        dataset  = loadmat(base_dir + '/usps_28x28.mat')
        data_set = dataset['dataset']
    
    img_train = data_set[0][0]
    label_train = data_set[0][1]
    img_test = data_set[1][0]
    label_test = data_set[1][1]
    inds = np.random.permutation(img_train.shape[0])
    img_train = img_train[inds]
    label_train = label_train[inds]
    
    img_train = img_train * 255
    img_test = img_test * 255
    img_train = img_train.reshape((img_train.shape[0], 1, 28, 28))
    img_test = img_test.reshape((img_test.shape[0], 1, 28, 28))

    #img_test = dense_to_one_hot(img_test)
    label_train = dense_to_one_hot(label_train)
    label_test = dense_to_one_hot(label_test)
    if usps_less_data_protocol:
        img_train = img_train[inds][:1800]
        label_train = label_train[inds][:1800]
    else:
        # Repeat trainset
        img_train = np.concatenate([img_train, img_train, img_train, img_train], 0)
        label_train = np.concatenate([label_train, label_train, label_train, label_train], 0)
    print('usps train X shape->',  img_train.shape)
    print('usps train y shape->',  label_train.shape)
    print('usps test X shape->',  img_test.shape)
    print('usps test y shape->', label_test.shape)


    return img_train, label_train, img_test, label_test
