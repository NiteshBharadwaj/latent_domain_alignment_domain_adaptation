#!/bin/bash

python3 main.py --target 'mnist_syn_mnistm_svhn_usps' --dl_type 'soft_cluster' --num_domain 5 --class_disc 'no' --record_folder '/results2/fresults/usps5d-sc0' --seed 0 --office_directory '/localdata/digits/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.001

#python3 main.py --target 'mnist_syn_mnistm_svhn_usps' --dl_type 'soft_cluster' --num_domain 5 --class_disc 'no' --record_folder '/results2/fresults/usps5d-sc1' --seed 1 --office_directory '/localdata/digits/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.001

#python3 main.py --target 'mnist_syn_mnistm_svhn_usps' --dl_type 'soft_cluster' --num_domain 5 --class_disc 'no' --record_folder '/results2/fresults/usps5d-sc2' --seed 2 --office_directory '/localdata/digits/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.001