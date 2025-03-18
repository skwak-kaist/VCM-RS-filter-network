#!/bin/bash

eval "$(conda shell.bash hook)"

env_name=$(head -n 1 ../vcm_env_name.txt | xargs)
echo $env_name
conda activate $env_name


python run_and_eval.py \
		--version v0.12 \
		--scenario inner \
		--component-order TSJRB_SRPTB \
		--dataset SFU \
		--gpu-ids 0,1 \
		--num-worker-per-gpu 6 \
		--configs RA,LD,AI \
		--seq-ids $(seq -s, 7) \
		--selection-algorithm yuv_pre \
		--pre-domain YUV \
		--post-domain RGB \
		--experiment-suffix pre_yuv_0.12 
		
