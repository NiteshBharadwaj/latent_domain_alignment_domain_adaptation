import os
# from tqdm import tqdm 

# dataset = ['mnistm', 'mnist', 'usps', 'svhn', 'syn']
# epochs = 5
# gpuid = 0
# experiments = ['hard_cluster','soft_cluster']
# k = 4

# #for data in tqdm(dataset):
#     #for exp in tqdm(experiments):
#         #for i in tqdm(range(5)):
#             #command = 'bash experiment_do.sh %s %d %d record/%s_%s_clus_%d %s digits %d' % (data,epochs,gpuid,data,exp,k,exp,k)
#             #print(command)
#             #os.system(command)
#             #bash experiment_do.sh  usps 100 0 record/usps_MSDA_{exp_type} digits 4

# k_list = [1,2,5,8,10]

# for data in tqdm(dataset):
#     for exp in tqdm(experiments):
#         for k in tqdm(k_list):
#             command = 'bash experiment_do.sh %s %d %d record/%s_%s_clus_%d %s digits %d' % (data,epochs,gpuid,data,exp,k,exp,k)
#             print(command)
#             os.system(command)

command = 'experiment_do.sh dslr 100 0 record/dslr_source_only soft_cluster office 2'
print(command)
os.system(command)
# data = 'dslr'
# epochs = 100
# gpuid = 0
# exp = 'soft_cluster'
# k = 2
# command = 'bash experiment_do.sh %s %d %d record/%s_%s_clus_%d %s digits %d' % (data,epochs,gpuid,data,exp,k,exp,k)
# print(command)
# os.system(command)