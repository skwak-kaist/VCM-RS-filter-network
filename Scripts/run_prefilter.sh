#!/bin/bash

eval "$(conda shell.bash hook)"

env_name=$(head -n 1 ../vcm_env_name.txt | xargs)
echo $env_name
conda activate $env_name

num_workers=12

python run_and_eval.py \
		--version v0.12 \
		--scenario e2e \
		--component-order TCSJRBD_SRCPTPB \
		--dataset SFU \
		--gpu-ids 0,1 \
		--num-worker-per-gpu ${num_workers} \
		--configs AI \
		--seq-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
		--selection-algorithm yuv_pre \
		--pre-domain YUV \
		--post-domain RGB \
		--experiment-suffix pre_yuv_0.12 
		

# python run_and_eval.py \
# 		--version v0.12 \
# 		--scenario inner \
# 		--component-order TCSJRBD_SRCPTPB \
# 		--dataset SFU \
# 		--gpu-ids 0,1 \
# 		--num-worker-per-gpu ${num_workers} \
# 		--configs RA,LD,AI \
# 		--seq-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
# 		--selection-algorithm yuv_pre \
# 		--pre-domain YUV \
# 		--post-domain RGB \
# 		--experiment-suffix pre_yuv_0.12 
		
# python run_and_eval.py \
# 		--version v0.12 \
# 		--scenario e2e \
# 		--component-order TCSJRBD_SRCPTPB \
# 		--dataset SFU \
# 		--gpu-ids 0,1 \
# 		--num-worker-per-gpu ${num_workers} \
# 		--configs RA,LD,AI \
# 		--seq-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
# 		--selection-algorithm yuv_pre \
# 		--pre-domain YUV \
# 		--post-domain RGB \
# 		--experiment-suffix pre_yuv_0.12 

# python run_and_eval.py \
# 		--version v0.12 \
# 		--scenario inner \
# 		--component-order TCSJRBD_SRCPTPB \
# 		--dataset TVD \
# 		--gpu-ids 0,1 \
# 		--num-worker-per-gpu ${num_workers} \
# 		--configs RA,LD,AI \
# 		--seq-ids 1,2,3,4,5,6,7 \
# 		--selection-algorithm yuv_pre \
# 		--pre-domain YUV \
# 		--post-domain RGB \
# 		--experiment-suffix pre_yuv_0.12 

# python run_and_eval.py \
# 		--version v0.12 \
# 		--scenario e2e \
# 		--component-order TCSJRBD_SRCPTPB \
# 		--dataset TVD \
# 		--gpu-ids 0,1 \
# 		--num-worker-per-gpu ${num_workers} \
# 		--configs RA,LD,AI \
# 		--seq-ids 1,2,3,4,5,6,7 \
# 		--selection-algorithm yuv_pre \
# 		--pre-domain YUV \
# 		--post-domain RGB \
# 		--experiment-suffix pre_yuv_0.12 