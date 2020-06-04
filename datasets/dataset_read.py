import sys

sys.path.append('../loader')
from unaligned_data_loader import UnalignedDataLoader
from .unaligned_data_loader_combined import UnalignedDataLoader as CombinedDataLoader
from .unaligned_data_loader_combined import UnalignedDataLoaderDomain as CombinedDataLoaderDomain
from class_wise_data_loader import ClasswiseDataLoader
from svhn import load_svhn
from mnist import load_mnist
from mnist_m import load_mnistm
from usps_ import load_usps
from gtsrb import load_gtsrb
from synth_number import load_syn
from synth_traffic import load_syntraffic

# User imports for hard-cluster code
import numpy as np
#import random
# from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def return_dataset(data, directory=".", is_multi=True, usps_less_data_protocol=False):
    use_full = not is_multi
    if data == 'svhn':
        train_image, train_label, \
        test_image, test_label = load_svhn(directory, use_full= use_full)
    if data == 'mnist':
        train_image, train_label, \
        test_image, test_label = load_mnist(directory, use_full=use_full, usps_less_data_protocol=usps_less_data_protocol)
        # print(train_image.shape)
    if data == 'mnistm':
        train_image, train_label, \
        test_image, test_label = load_mnistm(directory, use_full=use_full)
        # print(train_image.shape)
    if data == 'usps':
        train_image, train_label, \
        test_image, test_label = load_usps(directory, use_full=use_full, usps_less_data_protocol=usps_less_data_protocol)
    if data == 'synth':
        train_image, train_label, \
        test_image, test_label = load_syntraffic(directory)
    if data == 'gtsrb':
        train_image, train_label, \
        test_image, test_label = load_gtsrb(directory)
    if data == 'syn':
        train_image, train_label, \
        test_image, test_label = load_syn(directory)

    return train_image, train_label, test_image, test_label


def dataset_read(target, batch_size):
    S1 = {}
    S1_test = {}
    S2 = {}
    S2_test = {}
    S3 = {}
    S3_test = {}
    S4 = {}
    S4_test = {}

    S = [S1, S2, S3, S4]
    S_test = [S1_test, S2_test, S3_test, S4_test]

    T = {}
    T_test = {}
    T_val = {}
    domain_all = ['mnistm', 'mnist', 'usps', 'svhn', 'syn']
    domain_all.remove(target)

    target_train, target_train_label, target_test, target_test_label = return_dataset(target)
    indices_tar = np.arange(0, target_test.shape[0])

    np.random.seed(42)
    np.random.shuffle(indices_tar)
    val_split = int(0.05 * target_test.shape[0])
    target_val = target_test[indices_tar[:val_split]]
    target_val_label = target_test_label[indices_tar[:val_split]]
    target_test = target_test[indices_tar[val_split:]]
    target_test_label = target_test_label[indices_tar[val_split:]]
    import pdb
    pdb.set_trace()
    for i in range(len(domain_all)):
        source_train, source_train_label, source_test, source_test_label = return_dataset(domain_all[i])
        S[i]['imgs'] = source_train
        S[i]['labels'] = source_train_label
        # input target sample when test, source performance is not important
        S_test[i]['imgs'] = target_test
        S_test[i]['labels'] = target_test_label

    # S['imgs'] = train_source
    # S['labels'] = s_label_train
    T['imgs'] = target_train
    T['labels'] = target_train_label

    # input target samples for both
    # S_test['imgs'] = test_target
    # S_test['labels'] = t_label_test
    T_test['imgs'] = target_test
    T_test['labels'] = target_test_label

    T_val['imgs'] = target_val
    T_val['labels'] = target_val_label

    scale = 32

    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size, scale=scale)
    dataset = train_loader.load_data()

    test_loader = UnalignedDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, batch_size, scale=scale)

    dataset_test = test_loader.load_data()

    S_val = {}
    S_val['imgs'] = np.zeros((20, 3, 32, 32))
    S_val['labels'] = np.zeros((20))
    val_loader = UnalignedDataLoader()
    val_loader.initialize([S_val], T_val, batch_size, batch_size, scale=scale)

    dataset_valid = val_loader.load_data()
    return dataset, dataset_test, dataset_valid


def dataset_hard_cluster(target, batch_size, num_clus, directory, seed, usps_less_data_protocol=False):

    # Number of components for PCA
    n_comp = 50

    S = []
    S_test = []
    T = {}
    T_test = {}
    T_val = {}
    
    target = target.split('_')
    source_codes = target[:-1]
    target_code = target[-1]
    
    domain_map = {'mnistm':'mnistm', 'mnist':'mnist', 'usps':'usps', 'svhn':'svhn', 'syn':'syn'}
    
    if len(source_codes)<1 or source_codes[0]=='all':
        domain_all = ['mnistm', 'mnist', 'usps', 'svhn', 'syn']
        domain_all.remove(target_code)
        source_codes = domain_all
    target = domain_map[target_code]
    domain_all = [domain_map[x] for x in domain_map if x in source_codes]

    for i in range(len(domain_all)):
        S.append({})
        S_test.append({})
    is_multi =  len(domain_all) > 1   
    target_train, target_train_label, target_test, target_test_label = return_dataset(target, directory=directory, is_multi=is_multi, usps_less_data_protocol=usps_less_data_protocol)
    
    indices_tar = np.arange(0,target_test.shape[0])
    np.random.seed(seed)
    np.random.shuffle(indices_tar)
    target_test = target_test[indices_tar]
    target_test_label = target_test_label[indices_tar]
    n_images_per_class = 100
    if target=='usps':
        n_images_per_class = 10
    valid_mask = []
    for i in range(10):
        select_indices = np.where(target_test_label == i)[0][:n_images_per_class]
        valid_mask.extend(select_indices.tolist())
    test_mask = [i for i in range(target_test_label.shape[0]) if i not in valid_mask]
    target_val = target_test[valid_mask]
    target_val_label = target_test_label[valid_mask]
    target_test = target_test[test_mask]
    target_test_label = target_test_label[test_mask]
    # Generate target dataset label splits
    # target_train, target_train_label, target_test, target_test_label = return_dataset(target_dataset)
    
    S_train = []
    S_train_labels = []
    # S_train_std = []        

    # Read the respective source domain datasets
    for i in range(len(domain_all)):

        source_train, source_train_label, source_test, source_test_label = return_dataset(domain_all[i], directory=directory, is_multi=is_multi, usps_less_data_protocol=usps_less_data_protocol)
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

    T['imgs'] = target_train
    T['labels'] = target_train_label

    # input target samples for both
    T_test['imgs'] = target_test
    T_test['labels'] = target_test_label

    T_val['imgs'] = target_val
    T_val['labels'] = target_val_label
    usps_only_exp = False
    if is_multi: #Like Peng
        scale = 32 # Scale everything to 32
    else:
        # Traffic Signs are typically scaled to 40
        # USPS only is scaled to 28 (Src: Maximum classifier discrepancy)
        traffic_exp = 'synth' in source_codes
        usps_only_exp = target=='usps' or 'usps' in source_codes
        scale = 40 if traffic_exp else 28 if usps_only_exp else 32


    X_combined = np.concatenate(S_train, axis=0)
    X_labels = np.concatenate(S_train_labels, axis=0)
    source_num_train_ex = X_combined.shape[0]
    X_vec = X_combined.reshape(source_num_train_ex, -1)

    mean = X_vec.mean(0)
    X_vec = X_vec - mean
    pca_transformed = PCA(n_components=n_comp).fit_transform(X_vec)
    kmeans = KMeans(n_clusters=num_clus, n_init=1)
    predict = kmeans.fit(pca_transformed).predict(pca_transformed)

    print("Hard Clustering Ends")
    S = []
    S_test = []
    for i in range(num_clus):
        S.append({})
        S[i]['imgs'] = X_combined[predict == i]
        S[i]['labels'] = X_labels[predict == i]
        S[i]['domain_labels'] = [i] * len(S[i]['labels'])

        # input target sample when test, source performance is not important
        S_test.append({})
        S_test[i]['imgs'] = target_test
        S_test[i]['labels'] = target_test_label

    scale = 32

    train_loader = CombinedDataLoaderDomain()
    train_loader.initialize(S, T, batch_size, batch_size, scale=scale)
    dataset = train_loader.load_data()

    test_loader = UnalignedDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, batch_size, scale=scale)

    dataset_test = test_loader.load_data()

    S_val = {}
    S_val['imgs'] = np.zeros((20, 3, 32, 32))
    S_val['labels'] = np.zeros((20))
    val_loader = UnalignedDataLoader()
    val_loader.initialize([S_val], T_val, batch_size, batch_size, scale=scale)

    dataset_valid = val_loader.load_data()

    return dataset, dataset_test, dataset_valid, is_multi, usps_only_exp


# If multi-domain, read only a subset of data (as used by Peng.)
# For single domain, usps less data protocol use less data
# In other settings, use full dataset
def dataset_combined(target, batch_size, num_clus, directory, seed, usps_less_data_protocol=False):
    S = []
    S_test = []
    T = {}
    T_test = {}
    T_val = {}
    
    target = target.split('_')
    source_codes = target[:-1]
    target_code = target[-1]
    domain_map = {'mnistm':'mnistm', 'mnist':'mnist', 'usps':'usps', 'svhn':'svhn', 'syn':'syn'}
    if len(source_codes)>1 or source_codes[0]=='all':
        domain_all = ['mnistm', 'mnist', 'usps', 'svhn', 'syn']
        domain_all.remove(target_code)
        source_codes = domain_all
    target = domain_map[target_code]
    domain_all = [domain_map[x] for x in domain_map if x in source_codes]

    for i in range(len(domain_all)):
        S.append({})
        S_test.append({})
    is_multi =  len(domain_all) > 1   
    target_train, target_train_label, target_test, target_test_label = return_dataset(target, directory=directory, is_multi=is_multi, usps_less_data_protocol=usps_less_data_protocol)
    
    indices_tar = np.arange(0,target_test.shape[0])
    np.random.seed(seed)
    np.random.shuffle(indices_tar)
    target_test = target_test[indices_tar]
    target_test_label = target_test_label[indices_tar]
    n_images_per_class = 100
    if target=='usps':
        n_images_per_class = 10
    valid_mask = []
    for i in range(10):
        select_indices = np.where(target_test_label == i)[0][:n_images_per_class]
        valid_mask.extend(select_indices.tolist())
    test_mask = [i for i in range(target_test_label.shape[0]) if i not in valid_mask]
    target_val = target_test[valid_mask]
    target_val_label = target_test_label[valid_mask]
    target_test = target_test[test_mask]
    target_test_label = target_test_label[test_mask]
    for i in range(len(domain_all)):
        source_train, source_train_label, source_test, source_test_label = return_dataset(domain_all[i], directory=directory, is_multi=is_multi, usps_less_data_protocol=usps_less_data_protocol)
        S[i]['imgs'] = source_train
        S[i]['labels'] = source_train_label
        # input target sample when test, source performance is not important
        S_test[i]['imgs'] = target_test
        S_test[i]['labels'] = target_test_label

    T['imgs'] = target_train
    T['labels'] = target_train_label

    # input target samples for both
    T_test['imgs'] = target_test
    T_test['labels'] = target_test_label

    T_val['imgs'] = target_val
    T_val['labels'] = target_val_label
    usps_only_exp = False
    if is_multi: #Like Peng
        scale = 32 # Scale everything to 32
    else:
        # Traffic Signs are typically scaled to 40
        # USPS only is scaled to 28 (Src: Maximum classifier discrepancy)
        traffic_exp = 'synth' in source_codes
        usps_only_exp = target=='usps' or 'usps' in source_codes
        scale = 40 if traffic_exp else 28 if usps_only_exp else 32
    train_loader = CombinedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size, scale=scale)
    dataset = train_loader.load_data()

    class_loader = ClasswiseDataLoader()
    class_loader.initialize(S, batch_size, scale=scale)
    dataset_class = class_loader.load_data()

    test_loader = UnalignedDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, batch_size, scale=scale)

    dataset_test = test_loader.load_data()

    S_val = {}
    S_val['imgs'] = np.zeros((20, 3, 32, 32))
    S_val['labels'] = np.zeros((20))
    val_loader = UnalignedDataLoader()
    val_loader.initialize([S_val], T_val, batch_size, batch_size, scale=scale)

    dataset_valid = val_loader.load_data()
    return dataset, dataset_test, dataset_valid, dataset_class, is_multi, usps_only_exp
