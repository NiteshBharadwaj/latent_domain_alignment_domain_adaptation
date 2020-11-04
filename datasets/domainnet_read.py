import csv
import random
import numpy as np

num_valid_per_class = 0

#TODO - update the function 
# How to divide the train test split
def read_domainnet_domain(domain, domainnet_directory, is_target, seed_id):
    random.seed(seed_id)
    trainFile = domainnet_directory + '/txt/' + domain + '_train.txt'
    testFile = domainnet_directory + '/txt/' + domain + '_test.txt'
    #combinedFile = domainnet_directory + '/txt/' + domain +"_combined.txt"

    #with open(combinedFile, "wb") as outfile:
    #    with open(trainFile, "rb") as infile:
    #        outfile.write(infile.read())
    #    with open(testFile, "rb") as infile:
    #        outfile.write(infile.read())
    #    with open(validFile, "rb") as infile:
    #        outfile.write(infile.read())

    trainReader = csv.reader(open(trainFile, 'r'))
    testReader = csv.reader(open(testFile, 'r'))

    paths_train, labels_train, paths_test, labels_test, paths_valid, labels_valid = [], [], [], [], [], []

    num_train_examples = 0
    for row in trainReader:
        num_train_examples += 1
        paths_train.append(domainnet_directory + row[0].split()[0])
        labels_train.append(int(row[0].split()[1]))

    random.Random(42).shuffle(paths_train)
    random.Random(42).shuffle(labels_train)

    valid_size = 3*len(paths_train)//10
    paths_valid = paths_train[:valid_size]
    labels_valid = labels_train[:valid_size]

    paths_train = paths_train[valid_size:]
    labels_train = labels_train[valid_size:]

    num_test_examples = 0
    for row in testReader:
        num_test_examples += 1
        paths_test.append(domainnet_directory + row[0].split()[0])
        labels_test.append(int(row[0].split()[1]))

    assert(len(paths_train) == len(labels_train))
    assert(len(paths_valid) == len(labels_valid))
    assert(len(paths_test) == len(labels_test))

    print("Domain-----", domain)
    print("Train_num_images:", len(paths_train))
    print("Valid_num_images:", len(paths_valid))
    print("Test_num_images:", len(paths_test))

    # classwise_imgs = {}
    #
    # for row in validReader:
    #     if int(row[0].split()[1])-1 in classwise_imgs:
    #         classwise_imgs[int(row[0].split()[1])-1].append(domainnet_directory + row[0].split()[0])
    #     else:
    #         classwise_imgs[int(row[0].split()[1])-1] = [domainnet_directory + row[0].split()[0]]
    #
    # for label_idx in classwise_imgs:
    #     tmp_arr = list(np.arange(len(classwise_imgs[label_idx])))
    #     sampled_arr = random.sample(tmp_arr, num_valid_per_class)
    #     for sampled_idx in sampled_arr:
    #         paths_valid.append(classwise_imgs[label_idx][sampled_idx])
    #         labels_valid.append(label_idx)
    #
    # for row in testReader:
    #     img_name = domainnet_directory + row[0].split()[0]
    #     if not (img_name in paths_valid):
    #         paths_test.append(img_name)
    #         labels_test.append(int(row[0].split()[1])-1)

    return paths_train, labels_train, paths_test, labels_test, paths_valid, labels_valid
