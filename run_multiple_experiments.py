import subprocess
import os

dataset = ['mnistm', 'mnist', 'usps', 'svhn', 'syn']
epochs = ' 100 0 record/'
foldername = '_MSDA_soft_cluster'


for data in dataset:
    for i in range(5):
    #subprocess.call(['bash','experiment.sh '+ data + ' soft_cluster'])
    

        command = 'bash experiment_do.sh ' + data + epochs+data+foldername+ ' soft_cluster'
        print(command)
        os.system(command)
        #('bash experiment_do.sh ' + data + epochs+data+foldername+ ' soft_cluster')
