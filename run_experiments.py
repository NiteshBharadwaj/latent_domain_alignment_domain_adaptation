import os
from tqdm import tqdm 

dataset = ['mnistm', 'mnist', 'usps', 'svhn', 'syn']

epochs = 100
gpuid = 0
experiments = ['soft_cluster','hard_cluster']
#,'source_only']

k_list = [4]
#k_list = [2,4,5,6,8,10]

for k in tqdm(k_list):
    for exp in tqdm(experiments):
        for data in tqdm(dataset):
            command = 'bash experiment_do.sh %s %d %d record/%s_%s_clus_%d %s digits %d yes' % (data,epochs,gpuid,data,exp,k,exp,k)
            print(command)
            os.system(command)
            # bash experiment_do.sh mnistm 100 0 record/mnistm_baseline soft_cluster digits 4 yes
