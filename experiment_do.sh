#!/usr/bin/bash
# only need to define target
echo "\$1（Target）: $1"
echo "\$2（Epoch）: $2"
echo "\$3（GPUID）: $3"
echo "\$4（Record): $4"
echo "\$5（Type): $5"
echo "\$6 (Data): $6"
echo "\$7 (Num Domains): $7"
echo "\$8 (Kl Weight): $8"
echo "\$9 (Seed): $9"
echo "\$10 (Office Data Directory) : ${10}"
echo "\$11 (Eval only): ${11}"

TARGET=$1
EPOCH=$2
GPUID=$3
RECORD=$4
TYPE=$5
DATA=$6
NUM_DOMAIN=$7
KL_WEIGHT=$8
SEED=$9
OFFICE_DIRECTORY=${10}
EVAL_ONLY=${11}
#mkdir 'tmp/tmp'.$RECORD
for i in `seq 1 1`
do
	CUDA_VISIBLE_DEVICES=${GPUID} python3 main.py  --record_folder ${RECORD} --target ${TARGET}  --max_epoch ${EPOCH} --dl_type ${TYPE} --data ${DATA} --num_domain ${NUM_DOMAIN} --kl_wt ${KL_WEIGHT} --seed ${SEED} --office_directory ${OFFICE_DIRECTORY} --eval_only ${EVAL_ONLY}
done

#eg. ./experiment_do.sh  usps 100 0 record/usps_MSDA_{exp_type} digits 4
