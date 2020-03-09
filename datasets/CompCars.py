import os
base_dir = 'data_cropped'
import scipy.io as sio
import numpy as np
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
        for folder2 in os.listdir(os.path.join(image_folder,folder)):
            labels.append(int(folder2))
    return labels

def read_images(image_folder):
    imgs = []
    for folder in os.listdir(image_folder):
        for folder2 in os.listdir(os.path.join(image_folder,folder)):
            for folder3 in os.listdir(os.path.join(image_folder,folder,folder2)):
                for img in os.listdir(os.path.join(image_folder,folder,folder2,folder3)):
                    imgs.append(os.path.join(folder,folder2,folder3,img))
    return imgs

def read_comp_cars(target):
    label_map = {}
    if target=='CCWeb':
        main_dir =os.path.join(base_dir,target,'data')
        img_dir = os.path.join(main_dir,'image')
        label_dir = os.path.join(main_dir,'label')
        all_imgs = read_images(img_dir)
        rel_paths_train = all_imgs
        labels_train = [int(x.split('/')[1]) for x in rel_paths_train]
        label_map = rev_(np.unique(np.array(read_labels(img_dir)))) 
        
        labels_train = [label_map[i] for i in labels_train]
        print("train_num_images {} ,{}".format(target,len(labels_train)))
        
        paths_train = [os.path.join(img_dir,x) for x in rel_paths_train]
        rel_paths_test = all_imgs
        labels_test = [int(x.split('/')[1]) for x in rel_paths_test]

        labels_test = [label_map[i] for i in labels_test] 
        print("test_num_images {} ,{}".format(target,len(labels_test)))

        paths_test = [os.path.join(img_dir,x) for x in rel_paths_test]
    if target=='CCSurv':

        main_dir_tr =os.path.join(base_dir,'CCWeb','data')
        img_dir_tr = os.path.join(main_dir_tr,'image')
        #train_file_tr = os.path.join(main_dir_tr,'train_test_split','classification','train.txt')
        #label_dir_tr = os.path.join(main_dir_tr,'label')
        #rel_paths_train = split_(train_file_tr, label_dir_tr)
        #labels_train_tr = [int(x.split('/')[1]) for x in rel_paths_train]
        label_map = rev_(np.unique(np.array(read_labels(img_dir_tr))))

        main_dir =os.path.join(base_dir,target,'sv_data')
        img_dir = os.path.join(main_dir,'image')
        label_dir = os.path.join(main_dir,'label')
        train_file = os.path.join(main_dir,'train_surveillance.txt')
        test_file = os.path.join(main_dir,'test_surveillance.txt')
        mat_file = os.path.join(main_dir,'sv_make_model_name.mat')
        mat = sio.loadmat(mat_file)

        rel_paths_train = split_(train_file, label_dir)
        labels_train = [int(x.split('/')[0]) for x in rel_paths_train] 
        labels_train = [mat['sv_make_model_name'][x-1,2][0][0] for x in labels_train]


        labels_train = [label_map[i] for i in labels_train]
        print("train_num_images {} ,{}".format(target,len(labels_train)))
        paths_train = [os.path.join(img_dir,x) for x in rel_paths_train]

        rel_paths_test = split_(test_file, label_dir)
        labels_test = [int(x.split('/')[0]) for x in rel_paths_test] 
        labels_test = [mat['sv_make_model_name'][x-1,2][0][0] for x in labels_test]
        paths_test = [os.path.join(img_dir,x) for x in rel_paths_test]
        labels_test = [label_map[i] for i in labels_test]
        print("test_num_images {} ,{}".format(target,len(labels_test)))
    return paths_train, labels_train, paths_test, labels_test
     
