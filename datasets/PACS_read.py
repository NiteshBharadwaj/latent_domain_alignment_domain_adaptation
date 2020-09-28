import csv
import random
import numpy as np

num_valid_per_class = 0

def read_pacs_domain(domain, pacs_directory, is_target, seed_id):
    random.seed(seed_id)
    trainFile = pacs_directory + '/splits/' + domain + '_train_kfold.txt'
    testFile = pacs_directory + '/splits/' + domain + '_test_kfold.txt'
    validFile = pacs_directory + '/splits/' + domain + '_crossval_kfold.txt'
    combinedFile = pacs_directory + '/splits/' + domain +"combined.txt"

    #with open(combinedFile, "wb") as outfile:
    #    with open(trainFile, "rb") as infile:
    #        outfile.write(infile.read())
    #    with open(testFile, "rb") as infile:
    #        outfile.write(infile.read())
    #    with open(validFile, "rb") as infile:
    #        outfile.write(infile.read())

    trainReader = csv.reader(open(combinedFile, 'r'))
    testReader = csv.reader(open(combinedFile, 'r'))
    validReader = csv.reader(open(combinedFile, 'r'))

    paths_train, labels_train, paths_test, labels_test, paths_valid, labels_valid = [], [], [], [], [], []

    for row in trainReader:
        paths_train.append(pacs_directory + row[0].split()[0])
        labels_train.append(int(row[0].split()[1])-1)

    if 1:#not is_target:
        for row in testReader:
            paths_test.append(pacs_directory + row[0].split()[0])
            labels_test.append(int(row[0].split()[1])-1)

        for row in validReader:
            paths_valid.append(pacs_directory + row[0].split()[0])
            labels_valid.append(int(row[0].split()[1])-1)
    else:
        classwise_imgs = {}

        for row in validReader:
            if int(row[0].split()[1])-1 in classwise_imgs:
                classwise_imgs[int(row[0].split()[1])-1].append(pacs_directory + row[0].split()[0])
            else:
                classwise_imgs[int(row[0].split()[1])-1] = [pacs_directory + row[0].split()[0]]

        for label_idx in classwise_imgs:
            tmp_arr = list(np.arange(len(classwise_imgs[label_idx])))
            sampled_arr = random.sample(tmp_arr, num_valid_per_class)
            for sampled_idx in sampled_arr:
                paths_valid.append(classwise_imgs[label_idx][sampled_idx])
                labels_valid.append(label_idx)

        for row in testReader:
            img_name = pacs_directory + row[0].split()[0]
            if not (img_name in paths_valid):
                paths_test.append(img_name)
                labels_test.append(int(row[0].split()[1])-1)

    return paths_train, labels_train, paths_test, labels_test, paths_valid, labels_valid
