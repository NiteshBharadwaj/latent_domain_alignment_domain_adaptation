import sys

sys.path.append('../loader')
sys.path.append('./datasets')
from unaligned_data_loader_office import UnalignedDataLoader as CombinedDataLoader
from OfficeRead import read_office_domain
# User imports for hard-cluster code
import numpy as np
#import random
# from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch

def return_dataset(target):
    return read_office_domain(target)


def office_combined(target, batch_size):
    S1 = {}
    S1_test = {}
    S1_valid = {}

    S = [{},{}]
    S_test = [{},{}]
    S_valid = [{},{}]

#     S = [{}]
#     S_test = [{}]
#     S_valid = [{}]

    T = {}
    T_test = {}
    T_valid = {}
    
#     domain_all = ['amazon','dslr']
    domain_all = ['amazon','dslr','webcam']
    domain_all.remove(target)

    target_train, target_train_label, target_test, target_test_label, target_valid, target_valid_label = return_dataset(target)

    for i in range(len(domain_all)):
        source_train, source_train_label, source_test, source_test_label, source_valid, source_valid_label = return_dataset(domain_all[i])
        S[i]['imgs'] = source_train
        S[i]['labels'] = source_train_label
        # input target sample when test, source performance is not important
        S_test[i]['imgs'] = target_test
        S_test[i]['labels'] = target_test_label

        S_valid[i]['imgs'] = target_valid
        S_valid[i]['labels'] = target_valid_label

    T['imgs'] = target_train
    T['labels'] = target_train_label

    T_test['imgs'] = target_test
    T_test['labels'] = target_test_label

    T_valid['imgs'] = target_valid
    T_valid['labels'] = target_valid_label

    scale = 256#TODO

    train_loader = CombinedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size, scale=scale, split='Train')
    dataset = train_loader.load_data()
    for i in range(len(domain_all)):
        S_test[i]['imgs'] = [S_test[i]['imgs'][0]]
        S_test[i]['labels'] = [S_test[i]['labels'][0]]
    test_loader = CombinedDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, batch_size, scale=scale, split='Test')
    dataset_test = test_loader.load_data()

    valid_loader = CombinedDataLoader()
    valid_loader.initialize(S_test, T_valid, batch_size, batch_size, scale=scale, split='Test')
    dataset_valid = valid_loader.load_data()
    

    return dataset, dataset_test, dataset_valid