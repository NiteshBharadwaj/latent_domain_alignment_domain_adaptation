#!/bin/bash

python3 main.py --target 'mnist_usps_syn_svhn_mnistm' --dl_type 'soft_cluster' --num_domain 5 --class_disc 'yes' --record_folder '/results2/fresults/mnistm5d-sc0-cd' --seed 0 --office_directory '/localdata/digits/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.005

python3 main.py --target 'mnist_usps_syn_svhn_mnistm' --dl_type 'soft_cluster' --num_domain 5 --class_disc 'yes' --record_folder '/results2/fresults/mnistm5d-sc1-cd' --seed 1 --office_directory '/localdata/digits/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.005

python3 main.py --target 'mnist_usps_syn_svhn_mnistm' --dl_type 'soft_cluster' --num_domain 5 --class_disc 'yes' --record_folder '/results2/fresults/mnistm5d-sc2-cd' --seed 2 --office_directory '/localdata/digits/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.005
