import csv
import random
import numpy as np

num_valid_per_class = 0


def read_office_caltech_domain(domain, office_caltech_directory, is_target, seed_id):
    random.seed(seed_id)

    trainFile = office_caltech_directory + '/data/Office-Caltech10/' + domain + '_data.csv'
    testFile = office_caltech_directory + '/data/Office-Caltech10/' + domain + '_data.csv'
    #validFile = office_caltech_directory + '/data/Office-Caltech10/' + domain + '_data.csv'

    trainReader = csv.reader(open(trainFile, 'r'))
    testReader = csv.reader(open(testFile, 'r'))
    #validReader = csv.reader(open(validFile, 'r'))

    paths_train, labels_train, paths_test, labels_test, paths_valid, labels_valid = [], [], [], [], [], []

    for row in trainReader:
        paths_train.append(office_caltech_directory + row[2][1:])
        labels_train.append(int(row[1]))

    if not is_target:
        for row in testReader:
            paths_test.append(office_caltech_directory + row[2][1:])
            labels_test.append(int(row[1]))

        #for row in validReader:
        #    paths_valid.append(office_caltech_directory + row[2][1:])
        #    labels_valid.append(int(row[1]))
    else:
        classwise_imgs = {}

        #for row in validReader:
        #    if int(row[1]) in classwise_imgs:
        #        classwise_imgs[int(row[1])].append(office_caltech_directory + row[2][1:])
        #    else:
        #        classwise_imgs[int(row[1])] = [office_caltech_directory + row[2][1:]]

        #for label_idx in classwise_imgs:
        #    tmp_arr = list(np.arange(len(classwise_imgs[label_idx])))
        #    sampled_arr = random.sample(tmp_arr, num_valid_per_class)
        #    for sampled_idx in sampled_arr:
        #        paths_valid.append(classwise_imgs[label_idx][sampled_idx])
        #        labels_valid.append(label_idx)

        for row in testReader:
            img_name = office_caltech_directory + row[2][1:]
            if not (img_name in paths_valid):
                paths_test.append(img_name)
                labels_test.append(int(row[1]))

    return paths_train, labels_train, paths_test, labels_test#, paths_valid, labels_valid
