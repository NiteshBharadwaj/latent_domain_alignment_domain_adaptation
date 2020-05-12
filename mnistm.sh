#!/usr/bin/bash
python3 main.py --target 'mnistm' --max_epoch 350 --dl_type 'soft_cluster' --data 'digits' --num_domain 5 --class_disc 'yes' --record_folder 'record_office/mnistm_b_0.1_yes' --kl_wt 0.1 --seed $1 --office_directory '/localdata/digits/Digit-Five/'

python3 main.py --target 'mnistm' --max_epoch 350 --dl_type 'soft_cluster' --data 'digits' --num_domain 5 --class_disc 'no' --record_folder 'record_office/mnistm_b_0.1_no' --kl_wt 1.0 --seed $1 --office_directory '/localdata/digits/Digit-Five/'

python3 main.py --target 'mnistm' --max_epoch 350 --dl_type 'source_target_only' --data 'digits' --num_domain 5 --class_disc 'yes' --record_folder 'record_office/mnistm_b_st_yes' --kl_wt 0.1 --seed $1 --office_directory '/localdata/digits/Digit-Five/'

python3 main.py --target 'mnistm' --max_epoch 350 --dl_type 'source_target_only' --data 'digits' --num_domain 5 --class_disc 'no' --record_folder 'record_office/mnistm_b_st_no' --kl_wt 1.0 --seed $1 --office_directory '/localdata/digits/Digit-Five/'
