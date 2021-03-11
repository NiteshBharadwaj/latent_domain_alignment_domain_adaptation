import csv
import random
import numpy as np

file_path = {
        "i": "/vulcan-pvc1/ml_for_da_pan_base/dataset_list/bird31_ina_list_2017.txt", #NOT AVAILABLE
        "n": "/vulcan-pvc1/ml_for_da_pan_base/dataset_list/bird31_nabirds_list.txt",
        "c": "/vulcan-pvc1/ml_for_da_pan_base/dataset_list/bird31_cub2011.txt"
        #         "pai":"/vulcan-pvc1/ml_for_da_pan_base/dataset_list/cub200_drawing_20.txt",
        #         "cub":"/vulcan-pvc1/ml_for_da_pan_base/dataset_list/cub200_2011_20.txt"
    }
def read_birds_domain(domain, birds_directory, is_target, seed_id):
    random.seed(seed_id)
    combinedFile = file_path[domain]
    paths = []
    labels = []
    with open(combinedFile,"r") as f:
        lines = f.readlines()
        for line in lines:
            path_label = line.strip().split(" ")
            path = path_label[0]
            label = int(path_label[1])
            paths.append(path)
            labels.append(label)

    return paths, labels, paths[:], labels[:], paths[:], labels[:]
