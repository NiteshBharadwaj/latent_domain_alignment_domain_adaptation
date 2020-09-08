import csv
import random
import numpy as np

def read_pacs(domain, pacs_directory, is_target, seed_id):
    random.seed(seed_id)
    trainFile = pacs_directory + '/' + domain + '_train_kfold.txt'
    testFile = pacs_directory + '/' + domain + '_test_kfold.txt'
    validFile = pacs_directory + '/' + domain + '_crossval_kfold.txt'

    trainReader = csv.reader(open(trainFile, 'r'))
    testReader = csv.reader(open(testFile, 'r'))
    validReader = csv.reader(open(validFile, 'r'))

    paths_train, labels_train, paths_test, labels_test, paths_valid, labels_valid = [], [], [], [], [], []

    for row in trainReader:
        paths_train.append(pacs_directory + row[2][1:])
        labels_train.append(int(row[1]))

    for row in testReader:
        paths_test.append(pacs_directory + row[2][1:])
        labels_test.append(int(row[1]))

    for row in validReader:
        paths_valid.append(pacs_directory + row[2][1:])
        labels_valid.append(int(row[1]))

    return paths_train, labels_train, paths_test, labels_test, paths_valid, labels_valid
