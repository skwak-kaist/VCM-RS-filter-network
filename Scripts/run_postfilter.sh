#!/bin/bash

eval "$(conda shell.bash hook)"

env_name=$(head -n 1 ../vcm_env_name.txt | xargs)
echo $env_name
conda activate $env_name


python run_and_eval.py \
		--version v0.12 \
		--scenario inner \
		--component-order TCSRBD_JSRCPTPB \
		--dataset SFU \
		--gpu-ids 0,1 \
		--num-worker-per-gpu 6 \
		--configs RA,LD,AI \
		--seq-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
		--selection-algorithm rgb_post_vtm_multi_qp_finetuned \
		--pre-domain YUV \
		--post-domain RGB \
		--experiment-suffix pre_yuv_0.12 
		
python run_and_eval.py \
		--version v0.12 \
		--scenario e2e \
		--component-order TCSRBD_JSRCPTPB \
		--dataset SFU \
		--gpu-ids 0,1 \
		--num-worker-per-gpu 6 \
		--configs RA,LD,AI \
		--seq-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
		--selection-algorithm rgb_post_vtm_multi_qp_finetuned \
		--pre-domain YUV \
		--post-domain RGB \
		--experiment-suffix pre_yuv_0.12 

python run_and_eval.py \
		--version v0.12 \
		--scenario inner \
		--component-order TCSRBD_JSRCPTPB \
		--dataset TVD \
		--gpu-ids 0,1 \
		--num-worker-per-gpu 6 \
		--configs RA,LD,AI \
		--seq-ids 1,2,3,4,5,6,7 \
		--selection-algorithm rgb_post_vtm_multi_qp_finetuned \
		--pre-domain YUV \
		--post-domain RGB \
		--experiment-suffix pre_yuv_0.12 

python run_and_eval.py \
		--version v0.12 \
		--scenario e2e \
		--component-order TCSRBD_JSRCPTPB \
		--dataset TVD \
		--gpu-ids 0,1 \
		--num-worker-per-gpu 6 \
		--configs RA,LD,AI \
		--seq-ids 1,2,3,4,5,6,7 \
		--selection-algorithm rgb_post_vtm_multi_qp_finetuned \
		--pre-domain YUV \
		--post-domain RGB \
		--experiment-suffix pre_yuv_0.12 
