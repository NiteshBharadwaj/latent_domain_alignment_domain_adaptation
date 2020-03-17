import os
base_dir = 'data_cropped'
import scipy.io as sio
import numpy as np
from random import shuffle

def split_(train_file, label_dir):
    paths = []
    with open(train_file,'r') as f:
        for line in f:
#             label_path = os.path.join(label_dir,line)
            paths.append(line.split('\n')[0])
    return paths
def rev_(map):
    map2 = {}
    for k,v in enumerate(map):
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
        for folder2 in os.listdir(os.path.join(image_folder,folder)):
            for folder3 in os.listdir(os.path.join(image_folder,folder,folder2)):
                for img in os.listdir(os.path.join(image_folder,folder,folder2,folder3)):
                    imgs.append(os.path.join(folder,folder2,folder3,img))
    return imgs

def model_num_to_car_name(image_folder):
    map1 = {}
    for folder in os.listdir(image_folder):
        for folder2 in os.listdir(os.path.join(image_folder,folder)):
            map1[int(folder2)] = int(folder)
    return map1

def read_comp_cars(target):
    label_map = {}
    if target=='CCWeb':
        main_dir =os.path.join(base_dir,target,'data')
        img_dir = os.path.join(main_dir,'image')
        label_dir = os.path.join(main_dir,'label')
        all_imgs = read_images(img_dir)

        train_file = os.path.join(main_dir, 'train_test_split', 'classification', 'train.txt')
        test_file = os.path.join(main_dir, 'train_test_split', 'classification', 'test.txt')
        rel_paths_train = split_(train_file, label_dir)
        rel_paths_test = split_(test_file, label_dir)
        rel_paths_train_test = rel_paths_train + rel_paths_test
        rel_paths_train = rel_paths_train_test
        rel_paths_test = rel_paths_train_test
        
        labels_train = [int(x.split('/')[0]) for x in rel_paths_train]
        
        label_map = rev_(np.unique(np.array(read_labels(img_dir)))) 
        
        labels_train = [label_map[i] for i in labels_train]
        print("Num Train classes {}, {}, {}, {}".format(target, len(np.unique(np.array(labels_train))), min(labels_train), max(labels_train)))
        print("train_num_images {} ,{}".format(target,len(labels_train)))
        
        paths_train = [os.path.join(img_dir,x) for x in rel_paths_train]
        labels_test = [int(x.split('/')[0]) for x in rel_paths_test]

        labels_test = [label_map[i] for i in labels_test] 
        print("Num Test classes {}, {}".format(target, len(np.unique(np.array(labels_test))), min(labels_test), max(labels_test)))
        print("test_num_images {} ,{}".format(target,len(labels_test)))

        paths_test = [os.path.join(img_dir,x) for x in rel_paths_test]

        paths_valid = ['gg']
        labels_valid = ['lol']
        
    if target=='CCSurv':
        main_dir_tr = os.path.join(base_dir, 'CCWeb', 'data')
        img_dir_tr = os.path.join(main_dir_tr, 'image')
        label_map = rev_(np.unique(np.array(read_labels(img_dir_tr))))
        model_num_to_car_name_map = model_num_to_car_name(img_dir_tr)


        # Find source labels and create a list of labels in source
        train_file_tr = os.path.join(main_dir_tr,'train_test_split','classification','train.txt')
        test_file_tr = os.path.join(main_dir_tr,'train_test_split','classification','test.txt')
        label_dir_tr = os.path.join(main_dir_tr,'label')
        rel_paths_train_tr = split_(train_file_tr, label_dir_tr)
        rel_paths_test_tr = split_(test_file_tr, label_dir_tr)
        rel_paths_train_test_tr = rel_paths_train_tr + rel_paths_test_tr
        rel_paths_train_tr = rel_paths_train_test_tr
        
        labels_train_tr = [int(x.split('/')[0]) for x in rel_paths_train_tr]

        labels_train_tr = [label_map[i] for i in labels_train_tr]
        available_labels = set(np.unique(np.array(labels_train_tr)).flatten())

        main_dir =os.path.join(base_dir,target,'sv_data')
        img_dir = os.path.join(main_dir,'image')
        label_dir = os.path.join(main_dir,'label')
        train_file = os.path.join(main_dir,'train_surveillance.txt')
        test_file = os.path.join(main_dir,'test_surveillance.txt')
        mat_file = os.path.join(main_dir,'sv_make_model_name.mat')
        mat = sio.loadmat(mat_file)

        rel_paths_train = split_(train_file, label_dir)
        labels_train = [int(x.split('/')[0]) for x in rel_paths_train] 
        labels_train = [model_num_to_car_name_map[mat['sv_make_model_name'][x-1,2][0][0]] for x in labels_train]


        labels_train = [label_map[i] for i in labels_train]
        print("Num Train classes {}, {}, {}, {}".format(target, len(np.unique(np.array(labels_train))), min(labels_train), max(labels_train)))
        unav_train = set([i for i in range(len(labels_train)) if labels_train[i] not in available_labels])
        print("train_num_images {} ,{}".format(target,len(labels_train)))
        print("train_num_images with unknown label {} ,{}".format(target, len(unav_train)))
        paths_train = [os.path.join(img_dir,x) for x in rel_paths_train]

        paths_train_sh = [paths_train[i] for i in range(len(paths_train)) if i not in unav_train]
        labels_train_sh = [labels_train[i] for i in range(len(paths_train)) if i not in unav_train]

        rel_paths_test = split_(test_file, label_dir)
        shuffle(rel_paths_test)
        valid_size = 6*len(rel_paths_test)//10
        rel_paths_valid = rel_paths_test[:valid_size]
        rel_paths_test = rel_paths_test[valid_size:]


        labels_test = [int(x.split('/')[0]) for x in rel_paths_test] 
        labels_test = [model_num_to_car_name_map[mat['sv_make_model_name'][x-1,2][0][0]] for x in labels_test]
        paths_test = [os.path.join(img_dir,x) for x in rel_paths_test]
        labels_test = [label_map[i] for i in labels_test]
        print("Num Test classes {}, {}, {}, {}".format(target, len(np.unique(np.array(labels_test))), min(labels_test), max(labels_test)))
        unav_test = set([i for i in range(len(labels_test)) if labels_test[i] not in available_labels])
        print("test_num_images {} ,{}".format(target,len(paths_test)))
        print("test_num_images with unknown label {} ,{}".format(target, len(unav_test)))
        paths_test_sh = [paths_test[i] for i in range(len(paths_test)) if i not in unav_test]
        labels_test_sh = [labels_test[i] for i in range(len(paths_test)) if i not in unav_test]
        


        labels_valid = [int(x.split('/')[0]) for x in rel_paths_valid] 
        labels_valid = [model_num_to_car_name_map[mat['sv_make_model_name'][x-1,2][0][0]] for x in labels_valid]
        paths_valid = [os.path.join(img_dir,x) for x in rel_paths_valid]
        labels_valid = [label_map[i] for i in labels_valid]
        print("Num Valid classes {}, {}, {}, {}".format(target, len(np.unique(np.array(labels_valid))), min(labels_valid), max(labels_valid)))
        unav_valid = set([i for i in range(len(labels_valid)) if labels_valid[i] not in available_labels])
        print("valid_num_images {} ,{}".format(target,len(paths_valid)))
        print("valid_num_images with unknown label {} ,{}".format(target, len(unav_valid)))
        paths_valid_sh = [paths_valid[i] for i in range(len(paths_valid)) if i not in unav_valid]
        labels_valid_sh = [labels_valid[i] for i in range(len(paths_valid)) if i not in unav_valid]


        #paths_train = paths_train_sh
        paths_test = paths_test_sh
        paths_valid = paths_valid_sh
        #labels_train = labels_train_sh
        labels_test = labels_test_sh
        labels_valid = labels_valid_sh
        print("train_num_images final {} ,{}".format(target,len(labels_train)))
        print("test_num_images final {} ,{}".format(target, len(labels_test)))
        print("valid_num_images final {} ,{}".format(target, len(labels_valid)))

    return paths_train, labels_train, paths_test, labels_test, paths_valid, labels_valid
     
