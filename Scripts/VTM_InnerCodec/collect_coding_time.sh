#!/bin/bash

eval "$(conda shell.bash hook)"

env_name=$(head -n 1 ../../vcm_env_name.txt | xargs)
echo $env_name
conda activate $env_name

root=output
#root=/home/skwak/Workspace/Project_VCM/VCM-RS/Scripts/output
datasets="SFU"
encoding_modes="AI_e2e AI_inner LD_e2e LD_inner RA_e2e RA_inner"
#encoding_modes="AI_e2e"

#encoding_path=v0.12-TCSJRBD_SRCPTPB-pre_yuv_0.12
c#encoding_path=anchor

for dataset in $datasets; do

	for encoding_mode in $encoding_modes; do

		python collect_coding_time.py $root/${encoding_path}/${dataset}_${encoding_mode}/coding_log encoding
		echo "encoding time computed"
		
	done
done

decoding_path=decode_postfilter_vtm_bdr_off_re

for dataset in $datasets; do

	for encoding_mode in $encoding_modes; do

		python collect_coding_time.py $root/$decoding_path/${dataset}_${encoding_mode}/decoding_log decoding
		
	done
done

