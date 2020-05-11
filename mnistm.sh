#!/usr/bin/bash
python3 main.py --target 'mnistm' --max_epoch 350 --dl_type 'soft_cluster' --data 'digits' --num_domain 5 --class_disc 'yes' --record_folder 'record_office/mnistm_sc_nval' --kl_wt 0.0 --seed $1 --office_directory '/localdata/digits/Digit-Five/'

python3 main.py --target 'mnistm' --max_epoch 350 --dl_type 'source_target_only' --data 'digits' --num_domain 5 --class_disc 'yes' --record_folder 'record_office/fmnistm_st_nval' --kl_wt 1.0 --seed $1 --office_directory '/localdata/digits/Digit-Five/'

