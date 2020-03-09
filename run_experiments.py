import os

dataset = ['mnistm', 'mnist', 'usps', 'svhn', 'syn']
epochs = 100
gpuid = 0
experiments = ['hard_cluster','soft_cluster']
k = 4

for data in dataset:
    for exp in experiments:
        for i in range(5):
            command = 'bash experiment_do.sh %s %d %d record/%s_%s_clus_%d %s digits %d' % (data,epochs,gpuid,data,exp,k,exp,k)
            print(command)
            #os.system(command)
            #bash experiment_do.sh  usps 100 0 record/usps_MSDA_{exp_type} digits 4

k_list = [1,2,5,8,10]

for data in dataset:
    for exp in experiments:
        for k in k_list:
            command = 'bash experiment_do.sh %s %d %d record/%s_%s_clus_%d %s digits %d' % (data,epochs,gpuid,data,exp,k,exp,k)
            print(command)
            #os.system(command)

