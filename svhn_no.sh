python3 main.py --target 'svhn' --max_epoch 200 --dl_type 'soft_cluster' --data 'digits' --num_domain 4 --class_disc 'no' --record_folder 'record_office/svhn_0.0_no' --kl_wt 0.0 --seed 2 --office_directory '/localdata/digits/Digit-Five/'

python3 main.py --target 'svhn' --max_epoch 200 --dl_type 'source_target_only' --data 'digits' --num_domain 4 --class_disc 'no' --record_folder 'record_office/svhn_st_no' --kl_wt 1.0 --seed 2 --office_directory '/localdata/digits/Digit-Five/'


python3 main.py --target 'svhn' --max_epoch 200 --dl_type 'soft_cluster' --data 'digits' --num_domain 4 --class_disc 'no' --record_folder 'record_office/svhn_0.01_no' --kl_wt 0.01 --seed 2 --office_directory '/localdata/digits/Digit-Five/'
