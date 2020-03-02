import os
base_dir = 'data'
import scipy.io as sio
def split_(train_file, label_dir):
    paths = []
    with open(train_file,'r') as f:
        for line in f:
#             label_path = os.path.join(label_dir,line)
            paths.append(line)
    return paths

def read_comp_cars(target):
    if target=='CCWeb':
        main_dir =os.path.join(base_dir,target,'data')
        img_dir = os.path.join(main_dir,'image')
        label_dir = os.path.join(main_dir,'label')
        train_file = os.path.join(main_dir,'train_test_split','classification','train.txt')
        test_file = os.path.join(main_dir,'train_test_split','classification','test.txt')
        rel_paths_train = split_(train_file, label_dir)
        labels_train = [int(x.split('/')[1]) for x in rel_paths_train] 
        paths_train = [os.path.join(img_dir,x) for x in rel_paths_train]
        rel_paths_test = split_(test_file, label_dir)
        labels_test = [int(x.split('/')[1]) for x in rel_paths_test] 
        paths_test = [os.path.join(img_dir,x) for x in rel_paths_test]
    if target=='CCSurv':
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
        paths_train = [os.path.join(img_dir,x) for x in rel_paths_train]

        rel_paths_test = split_(test_file, label_dir)
        labels_test = [int(x.split('/')[0]) for x in rel_paths_test] 
        labels_test = [mat['sv_make_model_name'][x-1,2][0][0] for x in labels_test]
        paths_test = [os.path.join(img_dir,x) for x in rel_paths_test]
    return paths_train, labels_train, paths_test, labels_test
     
