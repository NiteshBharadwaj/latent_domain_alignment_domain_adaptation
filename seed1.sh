#python3 main.py --target 'ad' --max_epoch 2000 --dl_type 'source_target_only' --data 'office' --num_domain 4 --class_disc 'no' --record_folder 'record_office/ad_st' --kl_wt 1.0 --seed 1
#python3 main.py --target 'ad' --max_epoch 2000 --dl_type 'soft_cluster' --data 'office' --num_domain 4 --class_disc 'no' --record_folder 'record_office/ad_4_1.0' --kl_wt 1.0 --seed 1
#python3 main.py --target 'ad' --max_epoch 2000 --dl_type 'soft_cluster' --data 'office' --num_domain 8 --class_disc 'no' --record_folder 'record_office/ad_8_1.0' --kl_wt 1.0 --seed 1


#python3 main.py --target 'aw' --max_epoch 2000 --dl_type 'source_target_only' --data 'office' --num_domain 4 --class_disc 'no' --record_folder 'record_office/aw_st' --kl_wt 1.0 --seed 1
#python3 main.py --target 'aw' --max_epoch 2000 --dl_type 'soft_cluster' --data 'office' --num_domain 4 --class_disc 'no' --record_folder 'record_office/aw_4_1.0' --kl_wt 1.0 --seed 1
#python3 main.py --target 'aw' --max_epoch 2000 --dl_type 'soft_cluster' --data 'office' --num_domain 8 --class_disc 'no' --record_folder 'record_office/aw_8_1.0' --kl_wt 1.0 --seed 1


python3 main.py --target 'dwa' --dl_type 'soft_cluster' --num_domain 2 --class_disc 'no' --record_folder 'record_office_new/dwa_sc_2.000' --seed 0 --office_directory '.' --data 'office' --max_epoch 2000 --kl_wt 0.1 --entropy_wt 0.8 --to_detach 'yes' --msda_wt 0.1 

# python3 main.py --target 'da' --max_epoch 2000 --dl_type 'soft_cluster' --data 'office' --num_domain 4 --class_disc 'no' --record_folder 'record_office/da_4_1.0' --kl_wt 1.0 --seed 1
# python3 main.py --target 'da' --max_epoch 2000 --dl_type 'soft_cluster' --data 'office' --num_domain 8 --class_disc 'no' --record_folder 'record_office/da_8_1.0' --kl_wt 1.0 --seed 1



# python3 main.py --target 'dw' --max_epoch 2000 --dl_type 'source_target_only' --data 'office' --num_domain 4 --class_disc 'no' --record_folder 'record_office/dw_st' --kl_wt 1.0 --seed 1
# python3 main.py --target 'dw' --max_epoch 2000 --dl_type 'soft_cluster' --data 'office' --num_domain 4 --class_disc 'no' --record_folder 'record_office/dw_4_1.0' --kl_wt 1.0 --seed 1
# python3 main.py --target 'dw' --max_epoch 2000 --dl_type 'soft_cluster' --data 'office' --num_domain 8 --class_disc 'no' --record_folder 'record_office/dw_8_1.0' --kl_wt 1.0 --seed 1


# python3 main.py --target 'wa' --max_epoch 2000 --dl_type 'source_target_only' --data 'office' --num_domain 4 --class_disc 'no' --record_folder 'record_office/wa_st' --kl_wt 1.0 --seed 1
# python3 main.py --target 'wa' --max_epoch 2000 --dl_type 'soft_cluster' --data 'office' --num_domain 4 --class_disc 'no' --record_folder 'record_office/wa_4_1.0' --kl_wt 1.0 --seed 1
# python3 main.py --target 'wa' --max_epoch 2000 --dl_type 'soft_cluster' --data 'office' --num_domain 8 --class_disc 'no' --record_folder 'record_office/wa_8_1.0' --kl_wt 1.0 --seed 1


# python3 main.py --target 'wd' --max_epoch 2000 --dl_type 'source_target_only' --data 'office' --num_domain 4 --class_disc 'no' --record_folder 'record_office/wd_st' --kl_wt 1.0 --seed 1
# python3 main.py --target 'wd' --max_epoch 2000 --dl_type 'soft_cluster' --data 'office' --num_domain 4 --class_disc 'no' --record_folder 'record_office/wd_4_1.0' --kl_wt 1.0 --seed 1
# python3 main.py --target 'wd' --max_epoch 2000 --dl_type 'soft_cluster' --data 'office' --num_domain 8 --class_disc 'no' --record_folder 'record_office/wd_8_1.0' --kl_wt 1.0 --seed 1


# python3 main.py --target 'daw' --max_epoch 2000 --dl_type 'source_target_only' --data 'office' --num_domain 4 --class_disc 'no' --record_folder 'record_office/daw_st' --kl_wt 1.0 --seed 1
# python3 main.py --target 'daw' --max_epoch 2000 --dl_type 'soft_cluster' --data 'office' --num_domain 4 --class_disc 'no' --record_folder 'record_office/daw_4_1.0' --kl_wt 1.0 --seed 1
# python3 main.py --target 'daw' --max_epoch 2000 --dl_type 'soft_cluster' --data 'office' --num_domain 8 --class_disc 'no' --record_folder 'record_office/daw_8_1.0' --kl_wt 1.0 --seed 1


# python3 main.py --target 'dwa' --max_epoch 2000 --dl_type 'source_target_only' --data 'office' --num_domain 4 --class_disc 'no' --record_folder 'record_office/dwa_st' --kl_wt 1.0 --seed 1
# python3 main.py --target 'dwa' --max_epoch 2000 --dl_type 'soft_cluster' --data 'office' --num_domain 4 --class_disc 'no' --record_folder 'record_office/dwa_4_1.0' --kl_wt 1.0 --seed 1
# python3 main.py --target 'dwa' --max_epoch 2000 --dl_type 'soft_cluster' --data 'office' --num_domain 8 --class_disc 'no' --record_folder 'record_office/dwa_8_1.0' --kl_wt 1.0 --seed 1


# python3 main.py --target 'wad' --max_epoch 2000 --dl_type 'source_target_only' --data 'office' --num_domain 4 --class_disc 'no' --record_folder 'record_office/wad_st' --kl_wt 1.0 --seed 1
# python3 main.py --target 'wad' --max_epoch 2000 --dl_type 'soft_cluster' --data 'office' --num_domain 4 --class_disc 'no' --record_folder 'record_office/wad_4_1.0' --kl_wt 1.0 --seed 1
# python3 main.py --target 'wad' --max_epoch 2000 --dl_type 'soft_cluster' --data 'office' --num_domain 8 --class_disc 'no' --record_folder 'record_office/wad_8_1.0' --kl_wt 1.0 --seed 1

