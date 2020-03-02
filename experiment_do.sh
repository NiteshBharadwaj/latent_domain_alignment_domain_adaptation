#!/usr/bin/bash
# only need to define target
echo "\$1（Target）: $1"
echo "\$2（Epoch）: $2"
echo "\$3（GPUID）: $3"
echo "\$4（Record): $4"
echo "\$5（Type): $5"
echo "\$6 (Data): $6"

TARGET=$1
EPOCH=$2
GPUID=$3
RECORD=$4
TYPE=$5
DATA=$6
#mkdir 'tmp/tmp'.$RECORD
for i in `seq 1 1`
do
CUDA_VISIBLE_DEVICES=${GPUID} python3 main.py  --record_folder ${RECORD} --target ${TARGET}  --max_epoch ${EPOCH} --dl_type ${TYPE} --data ${DATA}
done

#eg. ./experiment_do.sh  usps 100 0 record/usps_MSDA_beta
# ./experiment_do.sh  svhn 100 0 record/svhn_MSDA_beta
# ./experiment_do.sh  mnistm 100 1 record/mnistm_MSDA_beta
# ./experiment_do.sh  mnist 100 1 record/mnist_MSDA_beta
# ./experiment_do.sh  syn 100 2 record/syn_MSDA_beta
