import sys
sys.path.append('../loader')

from .unaligned_data_loader import UnalignedDataLoader
from .svhn import load_svhn
from .mnist import load_mnist
from .mnist_m import load_mnistm
from .usps_ import load_usps
from .gtsrb import load_gtsrb
from .synth_number import load_syn
from .synth_traffic import load_syntraffic

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# from sklearn.preprocessing import StandardScaler

def return_dataset(data, scale=False, usps=False, all_use='no'):
    if data == 'svhn':
        train_image, train_label, test_image, test_label = load_svhn()
    if data == 'mnist':
        train_image, train_label, test_image, test_label = load_mnist()
    if data == 'mnistm':
        train_image, train_label, test_image, test_label = load_mnistm()
    if data == 'usps':
        train_image, train_label, test_image, test_label = load_usps()
    if data == 'synth':
        train_image, train_label, test_image, test_label = load_syntraffic()
    if data == 'gtsrb':
        train_image, train_label, test_image, test_label = load_gtsrb()
    if data == 'syn':
        train_image, train_label, test_image, test_label = load_syn()

    return train_image, train_label, test_image, test_label


def dataset_read(target, batch_size):
    num_latent_domains = 4
    scale = 32

    domain_all = ['mnistm', 'mnist', 'usps', 'svhn', 'syn']
    domain_all.remove(target)

    target_train, target_train_label, target_test, target_test_label = return_dataset(target)
    T = {'imgs': target_train, 'labels': target_train_label}

    # input target samples for both
    T_test = {'imgs': target_test, 'labels': target_test_label}

    S = {}
    S_test = {}
    for i in range(len(num_latent_domains)):
        source_train, source_train_label, source_test, source_test_label = return_dataset(domain_all[i])
        S[i] = {}
        S[i]['imgs'] = source_train
        S[i]['labels'] = source_train_label

        # input target sample when test, source performance is not important
        S_test[i] = {}
        S_test[i]['imgs'] = target_test
        S_test[i]['labels'] = target_test_label

    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size, scale=scale)
    dataset = train_loader.load_data()

    test_loader = UnalignedDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, batch_size, scale=scale)

    dataset_test = test_loader.load_data()

    return dataset, dataset_test


def dataset_hard_cluster(target, batch_size):
    # Number of components for PCA
    n_comp = 50
    # Number of hard clusters for K-Means algorithm
    num_clus = 7
    scale = 32

    domain_all = ['mnistm', 'mnist', 'usps', 'svhn', 'syn']
    domain_all.remove(target)
    target_train, target_train_label, target_test, target_test_label = return_dataset(target)

    T = {'imgs': target_train, 'labels': target_train_label}

    # input target samples for both
    T_test = {'imgs': target_test, 'labels': target_test_label}

    S_train = []
    S_train_labels = []
    # S_train_std = []

    # Read the respective source domain datasets
    for i in range(len(domain_all)):

        source_train, source_train_label, source_test, source_test_label = return_dataset(domain_all[i])
        # Convert all the datasets to (3,28,28) image size for (2352 Feature vector)
        # Broadcast to three channels
        if source_train.shape[1] == 1:
            source_train = np.repeat(source_train, 3, 1)
        # Clip to 28x28
        if source_train.shape[2] == 32:
            source_train = source_train[:, :, 2:30, 2:30]
        S_train.append(source_train)
        S_train_labels.append(source_train_label)

        # S_train_std.append(StandardScaler().fit_transform(source_train.reshape(source_train.shape[0], -1)))

    X_combined = np.concatenate(S_train, axis=0)
    X_labels = np.concatenate(S_train_labels, axis=0)
    source_num_train_ex = X_combined.shape[0]
    X_vec = X_combined.reshape(source_num_train_ex, -1)

    mean = X_vec.mean(0)
    X_vec = X_vec - mean
    pca_transformed = PCA(n_components=n_comp).fit_transform(X_vec)
    kmeans = KMeans(n_clusters=num_clus, n_init=1)
    predict = kmeans.fit(pca_transformed).predict(pca_transformed)

    S = {}
    S_test = {}
    for i in range(len(num_clus)):
        S[i] = {}
        S[i]['imgs'] = X_combined[predict == i]
        S[i]['labels'] = X_labels[predict == i]

        # input target sample when test, source performance is not important
        S_test[i] = {}
        S_test[i]['imgs'] = target_test
        S_test[i]['labels'] = target_test_label

    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size, scale=scale)
    dataset = train_loader.load_data()

    test_loader = UnalignedDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, batch_size, scale=scale)

    dataset_test = test_loader.load_data()

    return dataset, dataset_test
