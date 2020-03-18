import os

data = ['svhn','syn']

epochs = ' 100 0 record/'
folder = '_MSDA_hard_cluster'

for d in data:
    for i in range(5):
        command = 'bash experiment_do.sh ' + d + epochs + d + folder + ' hard_cluster digits'
        #print(command)
        os.system(command)

