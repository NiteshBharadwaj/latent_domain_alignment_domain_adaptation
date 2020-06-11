python3 main.py --target 'syn_svhn' --dl_type 'soft_cluster' --num_domain 2 --class_disc 'no' --record_folder '/results2/fresults/syn-svhn2d00005-nocd-sc0' --seed 0 --office_directory '/localdata/digits/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.0005

python3 main.py --target 'syn_svhn' --dl_type 'soft_cluster' --num_domain 2 --class_disc 'no' --record_folder '/results2/fresults/syn-svhn2d00005-nocd-sc1' --seed 1 --office_directory '/localdata/digits/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.0005

python3 main.py --target 'syn_svhn' --dl_type 'soft_cluster' --num_domain 2 --class_disc 'no' --record_folder '/results2/fresults/syn-svhn2d00005-nocd-sc2' --seed 2 --office_directory '/localdata/digits/Digit-Five' --data 'digits' --max_epoch 400 --kl_wt 0.01 --entropy_wt 0.01 --to_detach 'yes' --msda_wt 0.0005
