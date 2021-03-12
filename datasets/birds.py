import sys

sys.path.append('../loader')
sys.path.append('./datasets')
from unaligned_data_loader_birds import UnalignedDataLoader as CombinedDataLoader
from test_loader_birds import TestDataLoader as TestDataLoader
from BIRDS_read import read_birds_domain

from class_wise_dataloader_birds import ClasswiseDataLoader


def return_dataset(target, birds_directory, is_target, seed_id):
    return read_birds_domain(target, birds_directory, is_target, seed_id)

from image_list import ImageList
import pre_process as prep
import torch
def birds_combined(target, batch_size, birds_directory, seed_id, num_workers):
    T = {}
    T_test = {}
    T_valid = {}

    domain_dict = {'i': 'i', 'n': 'n', 'c': 'c'}
    target_domain = domain_dict[target[-1]]
    source_domains = target[:-1]
    domain_all = []
    for char in source_domains:
        domain_all.append(domain_dict[char])

    S = []
    S_test = []
    S_valid = []

    for i in range(len(domain_all)):
        S.append({})
        S_test.append({})
        S_valid.append({})

    target_train, target_train_label, target_test, target_test_label, target_valid, target_valid_label = return_dataset(
        target_domain, birds_directory, True, seed_id)

    for i in range(len(domain_all)):
        source_train, source_train_label, source_test, source_test_label, source_valid, source_valid_label = return_dataset(domain_all[i], birds_directory, False, seed_id)

        S[i]['imgs'] = source_train
        S[i]['labels'] = source_train_label
        # input target sample when test, source performance is not important
        S_test[i]['imgs'] = target_test
        S_test[i]['labels'] = target_test_label
        print('Loading from', domain_all[i])
        print('Train DataLoader Source of Size:', len(S[i]['labels']))
        S_valid[i]['imgs'] = target_valid
        S_valid[i]['labels'] = target_valid_label

    T['imgs'] = target_train
    T['labels'] = target_train_label

    T_test['imgs'] = target_test
    T_test['labels'] = target_test_label

    T_valid['imgs'] = target_valid
    T_valid['labels'] = target_valid_label

    scale = 256  # TODO

    train_loader = CombinedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size, num_workers, scale=scale, split='Train')
    dataset = train_loader.load_data()
    for i in range(len(domain_all)):
        S_test[i]['imgs'] = [S_test[i]['imgs'][0]]
        S_test[i]['labels'] = [S_test[i]['labels'][0]]

    test_loader = TestDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, batch_size, num_workers, scale=scale, split='Test')
    dataset_test = test_loader.load_data()
    print('Loading from', target_domain)
    print('Test DataLoader of size:', len(T_test['labels']))
    print('Train DataLoader Target of size:', len(T['labels']))
    print('Validation DataLoader of size:', len(T_valid['labels']))
    valid_loader = TestDataLoader()
    valid_loader.initialize(S_test, T_valid, batch_size, batch_size, num_workers, scale=scale, split='Test')
    dataset_valid = valid_loader.load_data()

    class_loader = ClasswiseDataLoader()
    class_loader.initialize(S, batch_size, num_workers, scale=scale)
    dataset_class = class_loader.load_data()

    file_path = {
        "i": "/vulcan-pvc1/ml_for_da_pan_base/dataset_list/bird31_ina_list_2017.txt",  # NOT AVAILABLE
        "n": "/vulcan-pvc1/ml_for_da_pan_base/dataset_list/bird31_nabirds_list.txt",
        "c": "/vulcan-pvc1/ml_for_da_pan_base/dataset_list/bird31_cub2011.txt"
        #         "pai":"/vulcan-pvc1/ml_for_da_pan_base/dataset_list/cub200_drawing_20.txt",
        #         "cub":"/vulcan-pvc1/ml_for_da_pan_base/dataset_list/cub200_2011_20.txt"
    }
    target_file = file_path[target[-1]]
    prep_dict_test = prep.image_test_10crop(resize_size=256, crop_size=224)
    valid_dataset_loaders = {}
    for i in range(10):
        dataset_list = ImageList(open(target_file).readlines(), transform=prep_dict_test["val" + str(i)],seed=seed_id, valid=True)
        valid_dataset_loaders["val" + str(i)] = torch.utils.data.DataLoader(dataset_list, batch_size=4, shuffle=False, num_workers=4)

    test_dataset_loaders = {}
    for i in range(10):
        dataset_list = ImageList(open(target_file).readlines(), transform=prep_dict_test["val" + str(i)],seed=seed_id, valid=False)
        test_dataset_loaders["val" + str(i)] = torch.utils.data.DataLoader(dataset_list, batch_size=4, shuffle=False, num_workers=4)
    return dataset, test_dataset_loaders, valid_dataset_loaders, dataset_class
