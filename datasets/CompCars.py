import os

#TODO: Add this in main file
#compcars_directory = 'data_cropped'
import scipy.io as sio
import numpy as np
import random


def split_(train_file, label_dir):
    paths = []
    with open(train_file, 'r') as f:
        for line in f:
            #             label_path = os.path.join(label_dir,line)
            paths.append(line.split('\n')[0])
    return paths


def rev_(map):
    map2 = {}
    for k, v in enumerate(map):
        map2[v] = k
    return map2


def read_labels(image_folder):
    labels = []
    for folder in os.listdir(image_folder):
        labels.append(int(folder))
        # for folder2 in os.listdir(os.path.join(image_folder,folder)):
        #     labels.append(int(folder2))
    return labels


def read_images(image_folder):
    imgs = []
    for folder in os.listdir(image_folder):
        for folder2 in os.listdir(os.path.join(image_folder, folder)):
            for folder3 in os.listdir(os.path.join(image_folder, folder, folder2)):
                for img in os.listdir(os.path.join(image_folder, folder, folder2, folder3)):
                    imgs.append(os.path.join(folder, folder2, folder3, img))
    return imgs


def model_num_to_car_name(image_folder):
    map1 = {}
    for folder in os.listdir(image_folder):
        for folder2 in os.listdir(os.path.join(image_folder, folder)):
            map1[int(folder2)] = int(folder)
    return map1

def get_aux_labels(file_name1, file_name2, label_dir):
     features = []
     with open(file_name1,'r') as f:
         for line in f:
             feature_file = os.path.join(label_dir, line.split('.')[0]+'.txt')
             tmp = [0]*2
             idx = 0
             with open(feature_file, 'r') as f2:
                 for line2 in f2:
                     tmp[idx] = int(line2.split('\n')[0])
                     idx += 1
                     if idx==2:
                         break
                     # tmp.append(line2.split('\n')[0])
             features.append(tmp)
     with open(file_name2,'r') as f:
         for line in f:
             feature_file = os.path.join(label_dir, line.split('.')[0]+'.txt')
             tmp = [0]*2
             idx = 0
             with open(feature_file, 'r') as f2:
                 for line2 in f2:
                     tmp[idx] = int(line2.split('\n')[0])
                     idx += 1
                     if idx==2:
                         break
                     # tmp.append(line2.split('\n')[0])
             features.append(tmp)
     return features

def read_comp_cars(target, compcars_directory, is_target, seed_id):
    # Returns train, test and validation images and labels for source or target domain
    #TODO is_target figure out usage
    label_map = {}
    random.seed(seed_id)

    if target == 'CCWeb':
        main_dir = os.path.join(compcars_directory, target, 'data')
        img_dir = os.path.join(main_dir, 'image')
        label_dir = os.path.join(main_dir, 'label')
        all_imgs = read_images(img_dir)

        train_file = os.path.join(main_dir, 'train_test_split', 'classification', 'train.txt')
        test_file = os.path.join(main_dir, 'train_test_split', 'classification', 'test.txt')
        rel_paths_train = split_(train_file, label_dir)
        rel_paths_test = split_(test_file, label_dir)
        rel_paths_train_test = rel_paths_train + rel_paths_test
        rel_paths_train = rel_paths_train_test
        rel_paths_test = rel_paths_train_test

        labels_train = [int(x.split('/')[0]) for x in rel_paths_train]

        label_map = rev_(np.unique(np.array(labels_train)))

        labels_train = [label_map[i] for i in labels_train]
        print(
            "Num Train classes {}, {}, {}, {}".format(target, len(np.unique(np.array(labels_train))), min(labels_train),
                                                      max(labels_train)))
        print("train_num_images {} ,{}".format(target, len(labels_train)))

        # -------------------------------------
        aux_labels_train = get_aux_labels(train_file, test_file, label_dir)
        assert(len(aux_labels_train) == len(labels_train))
        labels_train_tmp = []
        for i in range(len(labels_train)):
            labels_train_tmp.append([labels_train[i], aux_labels_train[i][0], aux_labels_train[i][1]])
        labels_train = labels_train_tmp
         # -------------------------------------

        paths_train = [os.path.join(img_dir, x) for x in rel_paths_train]
        labels_test = [int(x.split('/')[0]) for x in rel_paths_test]

        labels_test = [label_map[i] for i in labels_test]
        print("Num Test classes {}, {}".format(target, len(np.unique(np.array(labels_test))), min(labels_test),
                                               max(labels_test)))
        print("test_num_images {} ,{}".format(target, len(labels_test)))

         # -------------------------------------
        aux_labels_test = get_aux_labels(train_file, test_file, label_dir)
        assert(len(aux_labels_test) == len(labels_test))
        labels_test_tmp = []
        for i in range(len(labels_test)):
            labels_test_tmp.append([labels_test[i], aux_labels_test[i][0], aux_labels_test[i][1]])
        labels_test = labels_test_tmp
         # -------------------------------------

        paths_test = [os.path.join(img_dir, x) for x in rel_paths_test]

        paths_valid = [paths_test[0]]
        labels_valid = [labels_test[0]]

    if target == 'CCSurv':
        main_dir_tr = os.path.join(compcars_directory, 'CCWeb', 'data')
        img_dir_tr = os.path.join(main_dir_tr, 'image')

        model_num_to_car_name_map = model_num_to_car_name(img_dir_tr)

        # Find source labels and create a list of labels in source
        train_file_tr = os.path.join(main_dir_tr, 'train_test_split', 'classification', 'train.txt')
        test_file_tr = os.path.join(main_dir_tr, 'train_test_split', 'classification', 'test.txt')
        label_dir_tr = os.path.join(main_dir_tr, 'label')
        rel_paths_train_tr = split_(train_file_tr, label_dir_tr)
        rel_paths_test_tr = split_(test_file_tr, label_dir_tr)
        rel_paths_train_test_tr = rel_paths_train_tr + rel_paths_test_tr
        rel_paths_train_tr = rel_paths_train_test_tr

        labels_train_tr = [int(x.split('/')[0]) for x in rel_paths_train_tr]
        
        
        label_map = rev_(np.unique(np.array(labels_train_tr)))


        labels_train_tr = [label_map[i] for i in labels_train_tr]
        available_labels = set(np.unique(np.array(labels_train_tr)).flatten())

        main_dir = os.path.join(compcars_directory, target, 'sv_data')
        img_dir = os.path.join(main_dir, 'image')
        label_dir = os.path.join(main_dir, 'label')
        train_file = os.path.join(main_dir, 'train_surveillance.txt')
        test_file = os.path.join(main_dir, 'test_surveillance.txt')
        mat_file = os.path.join(main_dir, 'sv_make_model_name.mat')
        mat = sio.loadmat(mat_file)

        rel_paths_train = split_(train_file, label_dir)
        paths_train = [os.path.join(img_dir, x) for x in rel_paths_train]
        labels_train = [int(x.split('/')[0]) for x in rel_paths_train]
        labels_train = [model_num_to_car_name_map[mat['sv_make_model_name'][x - 1, 2][0][0]] for x in labels_train]

        unav_train = set([i for i in range(len(labels_train)) if labels_train[i] not in label_map])
        paths_train = [paths_train[i] for i in range(len(paths_train)) if i not in unav_train]
        labels_train = [label_map[labels_train[i]] for i in range(len(labels_train)) if i not in unav_train]
        print(
            "Num Train classes {}, {}, {}, {}".format(target, len(np.unique(np.array(labels_train))), min(labels_train),
                                                      max(labels_train)))
        print("using train_num_images {} ,{}".format(target, len(labels_train)))
        print("not train_num_images with unknown label {} ,{}".format(target, len(unav_train)))
        # -------------------------------------
        labels_train_tmp = [[i,-1,-1] for i in labels_train]
        labels_train = labels_train_tmp
        # -------------------------------------

        rel_paths_test = split_(test_file, label_dir)
        rel_paths_test.sort()
        random.Random(seed_id).shuffle(rel_paths_test)

        paths_test = [os.path.join(img_dir, x) for x in rel_paths_test]
        labels_test = np.array([int(x.split('/')[0]) for x in rel_paths_test])
        labels_test = [model_num_to_car_name_map[mat['sv_make_model_name'][x - 1, 2][0][0]] for x in labels_test]
        unav_test = set([i for i in range(len(labels_test)) if labels_test[i] not in label_map])
        
        
        paths_test = [paths_test[i] for i in range(len(paths_test)) if i not in unav_test]
        labels_test = np.array([label_map[labels_test[i]] for i in range(len(labels_test)) if i not in unav_test])

        n_images_per_class = 10
        valid_mask = []
        for i in range(75):
            select_indices = np.where(labels_test==i)[0][:n_images_per_class]
            valid_mask.extend(select_indices.tolist())
        test_mask = [i for i in range(len(labels_test)) if i not in valid_mask]
        paths_valid = [paths_test[i] for i in valid_mask]
        labels_valid = [labels_test[i] for i in valid_mask]
        paths_test = [paths_test[i] for i in test_mask]
        labels_test = [labels_test[i] for i in test_mask]


        print("not using test set images unknown label {} ,{}".format(target, len(unav_test)))
        print("Num Test classes test split {}, {}, {}, {}".format(target, len(np.unique(np.array(labels_test))), min(labels_test),
                                                       max(labels_test)))
        print("using test_num_images test split {} ,{}".format(target, len(labels_test)))

        print("Num valid classes valid split {}, {}, {}, {}".format(target, len(np.unique(np.array(labels_valid))), min(labels_valid),
                                                       max(labels_valid)))
        print("using valid_num_images valid split {} ,{}".format(target, len(labels_valid)))
        # -------------------------------------
        labels_test_tmp = [[i,-1,-1] for i in labels_test]
        labels_test = labels_test_tmp
        # -------------------------------------

        # -------------------------------------
        labels_valid_tmp = [[i,-1,-1] for i in labels_valid]
        labels_valid = labels_valid_tmp
        # -------------------------------------


    return paths_train, labels_train, paths_test, labels_test, paths_valid, labels_valid
