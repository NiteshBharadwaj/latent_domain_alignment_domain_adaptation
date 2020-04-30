#import os
#from tqdm import tqdm 
#
#dataset = ['mnistm', 'mnist', 'usps', 'svhn', 'syn']
#
#epochs = 100
#gpuid = 0
#experiments = ['soft_cluster','hard_cluster']
#
#k_list = [4,5,6,8,10]
#
#for k in tqdm(k_list):
#    for exp in tqdm(experiments):
#        for data in tqdm(dataset):
#            command = 'bash experiment_do.sh %s %d %d record/%s_%s_clus_%d %s digits %d yes' % (data,epochs,gpuid,data,exp,k,exp,k)
#            print(command)
#            os.system(command)
#            # bash experiment_do.sh mnistm 100 0 record/mnistm_baseline soft_cluster digits 4 yes


gpuid = 0
experiments1 = ['source_only','source_target']

experiments = ['soft_cluster']


epochs = {'source_only': 500,'source_target':1500,'soft_cluster':1500}

datasets = {'ad': 'amazon_dslr', 'aw': 'amazon_webcam', 'da': 'dslr_amazon', 'dw': 'dslr_webcam', 'wa': 'webcam_amazon', 'wd': 'webcam_dslr','daw': 'dslr_amazon_webcam', 'dwa': 'dslr_webcam_amazon', 'wad': 'webcam_amazon_dslr'}

k_list = [4,12,2,8,6]

exp_list = []

for k in k_list:
  for data in datasets:
    for exp in experiments:
      command = 'bash ./experiment_do.sh %s %d %d record_office/%s_%s_%d %s office %d no' % (data,epochs[exp],gpuid,datasets[data],exp,k,exp,k)
      #print(command)
      exp_list.append(command)
      #command = 'bash ./experiment_do.sh %s %d %d record_office/%s_%s %s office %d no' % (data,epochs[exp],gpuid,datasets[data],exp,exp,k)

for a in exp_list:
  print(a)
