#!/bin/bash

python3 main.py --target 'mnistm_svhn_usps_syn_mnist' --dl_type 'soft_cluster' --num_domain 5 --class_disc 'yes' --record_folder '/results2/fresults/mnist5d-so0-cd' --seed 0 --office_directory '/localdata/digits/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.0

python3 main.py --target 'mnistm_svhn_usps_syn_mnist' --dl_type 'soft_cluster' --num_domain 5 --class_disc 'yes' --record_folder '/results2/fresults/mnist5d-so1-cd' --seed 1 --office_directory '/localdata/digits/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.0

python3 main.py --target 'mnistm_svhn_usps_syn_mnist' --dl_type 'soft_cluster' --num_domain 5 --class_disc 'yes' --record_folder '/results2/fresults/mnist5d-so2-cd' --seed 2 --office_directory '/localdata/digits/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.0
