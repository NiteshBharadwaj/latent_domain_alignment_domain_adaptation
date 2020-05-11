python3 main.py --target 'mnistm' --max_epoch 100 --dl_type 'soft_cluster' --data 'digits' --num_domain 5 --class_disc 'yes' --record_folder 'record_office/mnistm_0.0_f2' --kl_wt 0.0 --seed 1 --office_directory '/localdata/digits/Digit-Five/'

python3 main.py --target 'mnistm' --max_epoch 100 --dl_type 'source_target_only' --data 'digits' --num_domain 5 --class_disc 'yes' --record_folder 'record_office/mnistm_st_f2' --kl_wt 1.0 --seed 1 --office_directory '/localdata/digits/Digit-Five/'


python3 main.py --target 'mnistm' --max_epoch 100 --dl_type 'soft_cluster' --data 'digits' --num_domain 5 --class_disc 'yes' --record_folder 'record_office/mnistm_0.01_f2' --kl_wt 0.01 --seed 1 --office_directory '/localdata/digits/Digit-Five/'
