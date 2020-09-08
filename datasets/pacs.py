import sys

sys.path.append('../loader')
sys.path.append('./datasets')
from unaligned_data_loader_pacs import UnalignedDataLoader as CombinedDataLoader
from test_loader_pacs import TestDataLoader as TestDataLoader
from PACS_read import read_pacs_domain

from class_wise_dataloader_pacs import ClasswiseDataLoader


def return_dataset(target, pacs_directory, is_target, seed_id):
    return read_pacs_domain(target, pacs_directory, is_target, seed_id)


def office_combined(target, batch_size, pacs_directory, seed_id, num_workers):
    S1 = {}
    S1_test = {}
    S1_valid = {}

    T = {}
    T_test = {}
    T_valid = {}

    domain_dict = {'p': 'photo', 'a': 'art_painting', 'c': 'cartoon', 's': 'sketch'}
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
        target_domain, pacs_directory, True, seed_id)

    for i in range(len(domain_all)):
        source_train, source_train_label, source_test, source_test_label, source_valid, source_valid_label = return_dataset(
            domain_all[i], pacs_directory, False, seed_id)

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

    print('Validation DataLoader of size:', len(T_valid['labels']))
    valid_loader = TestDataLoader()
    valid_loader.initialize(S_test, T_valid, batch_size, batch_size, num_workers, scale=scale, split='Test')
    dataset_valid = valid_loader.load_data()

    class_loader = ClasswiseDataLoader()
    class_loader.initialize(S, batch_size, num_workers, scale=scale)
    dataset_class = class_loader.load_data()

    return dataset, dataset_test, dataset_valid, dataset_class
