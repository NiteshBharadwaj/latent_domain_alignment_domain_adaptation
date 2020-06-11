#!/bin/bash

python3 main.py --target 'mnist_mnistm_svhn_usps_syn' --dl_type 'soft_cluster' --num_domain 3 --class_disc 'yes' --record_folder '/results2/fresults/syn3d-so0-cd' --seed 0 --office_directory '/localdata/digits/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.0

python3 main.py --target 'mnist_mnistm_svhn_usps_syn' --dl_type 'soft_cluster' --num_domain 3 --class_disc 'yes' --record_folder '/results2/fresults/syn3d-so1-cd' --seed 1 --office_directory '/localdata/digits/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.0

python3 main.py --target 'mnist_mnistm_svhn_usps_syn' --dl_type 'soft_cluster' --num_domain 3 --class_disc 'yes' --record_folder '/results2/fresults/syn3d-so2-cd' --seed 2 --office_directory '/localdata/digits/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.0
