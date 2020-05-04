#!/usr/bin/bash
# only need to define target
echo "\$1（Target）: $1"
echo "\$2（Epoch）: $2"
echo "\$3（GPUID）: $3"
echo "\$4（Record): $4"
echo "\$5（Type): $5"
echo "\$6 (Data): $6"
echo "\$7 (Num Domains): $7"
echo "\$8 (Seed): $8"
echo "\$9 (Eval only): $9"

TARGET=$1
EPOCH=$2
GPUID=$3
RECORD=$4
TYPE=$5
DATA=$6
NUM_DOMAIN=$7
SEED=$8
EVAL_ONLY=$9
#mkdir 'tmp/tmp'.$RECORD
for i in `seq 1 1`
do
	CUDA_VISIBLE_DEVICES=${GPUID} python3 main.py  --record_folder ${RECORD} --target ${TARGET}  --max_epoch ${EPOCH} --dl_type ${TYPE} --data ${DATA} --num_domain ${NUM_DOMAIN} --seed ${SEED} --eval_only ${EVAL_ONLY}
done

#eg. ./experiment_do.sh  usps 100 0 record/usps_MSDA_{exp_type} digits 4
