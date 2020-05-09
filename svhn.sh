python3 main.py --target 'svhn' --max_epoch 100 --dl_type 'soft_cluster' --data 'digits' --num_domain 4 --class_disc 'yes' --record_folder 'record_office/svhn_4_0.0' --kl_wt 0.0 --seed 1 --office_directory '/localdata/digits/Digit-Five/'

python3 main.py --target 'svhn' --max_epoch 100 --dl_type 'source_target_only' --data 'digits' --num_domain 4 --class_disc 'yes' --record_folder 'record_office/svhn_st' --kl_wt 1.0 --seed 1 --office_directory '/localdata/digits/Digit-Five/'
