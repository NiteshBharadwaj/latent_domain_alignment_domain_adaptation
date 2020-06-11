python3 main.py --target 'syn_svhn' --dl_type 'source_target_only' --num_domain 2 --class_disc 'no' --record_folder '/results2/fresults/syn-svhn2d-nocd-st0' --seed 0 --office_directory '/localdata/digits/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.001

python3 main.py --target 'syn_svhn' --dl_type 'source_target_only' --num_domain 2 --class_disc 'no' --record_folder '/results2/fresults/syn-svhn2d-nocd-st1' --seed 1 --office_directory '/localdata/digits/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.001

python3 main.py --target 'syn_svhn' --dl_type 'source_target_only' --num_domain 2 --class_disc 'no' --record_folder '/results2/fresults/syn-svhn2d-nocd-st2' --seed 2 --office_directory '/localdata/digits/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.001
