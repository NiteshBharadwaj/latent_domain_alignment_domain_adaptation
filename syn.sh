python3 main.py --target 'syn' --max_epoch 200 --dl_type 'soft_cluster' --data 'digits' --num_domain 2 --class_disc 'yes' --record_folder 'record_office/syn_0.0' --kl_wt 0.0 --seed 2 --office_directory '/localdata/digits/Digit-Five/'

python3 main.py --target 'syn' --max_epoch 200 --dl_type 'source_target_only' --data 'digits' --num_domain 2 --class_disc 'yes' --record_folder 'record_office/syn_st' --kl_wt 1.0 --seed 2 --office_directory '/localdata/digits/Digit-Five/'


python3 main.py --target 'syn' --max_epoch 200 --dl_type 'soft_cluster' --data 'digits' --num_domain 2 --class_disc 'yes' --record_folder 'record_office/syn_0.01' --kl_wt 0.01 --seed 2 --office_directory '/localdata/digits/Digit-Five/'
