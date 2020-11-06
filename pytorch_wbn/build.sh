#!/bin/bash

# Configuration
CUDA_GENCODE="\
-gencode=arch=compute_61,code=sm_61 \
-gencode=arch=compute_52,code=sm_52 \
-gencode=arch=compute_50,code=sm_50 \
-gencode=arch=compute_35,code=sm_35 \
-gencode=arch=compute_30,code=sm_30 \
-gencode=arch=compute_60,code=sm_60 "

cd src
nvcc -I/usr/local/cuda/include --expt-extended-lambda -O3 -c -o wbn.o wbn.cu -x cu -Xcompiler -fPIC -std=c++11 ${CUDA_GENCODE}
cd ..
