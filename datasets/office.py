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


def return_dataset(target):
    return read_comp_cars(target)


def office_combined(target, batch_size):
    S1 = {}
    S1_test = {}

    S = [{},{}]
    S_test = [{},{}]

    T = {}
    T_test = {}
    #domain_all = ['CCWeb', 'CCSurv']
    domain_all = ['amazon','dslr','webcam']
    domain_all.remove(target)

    target_train, target_train_label, target_test, target_test_label = read_office_domain(target)

    for i in range(len(domain_all)):
        source_train, source_train_label, source_test, source_test_label = read_office_domain(domain_all[i])
        S[i]['imgs'] = source_train
        S[i]['labels'] = source_train_label
        # input target sample when test, source performance is not important
        S_test[i]['imgs'] = target_test
        S_test[i]['labels'] = target_test_label

    T['imgs'] = target_train
    T['labels'] = target_train_label

    T_test['imgs'] = target_test
    T_test['labels'] = target_test_label

    scale = 256#TODO

    train_loader = CombinedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size, scale=scale, split='Train', shuffle_=True)
    dataset = train_loader.load_data()

    test_loader = CombinedDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, batch_size, scale=scale, split='Test', shuffle_=False)
    dataset_test = test_loader.load_data()

    # test_loader2 = CombinedDataLoader()
    # test_loader2.initialize(S_test, T_test, batch_size, batch_size, scale=scale, split='Test', shuffle_=False)
    # dataset_test2 = test_loader2.load_data()

    return dataset, dataset_test
