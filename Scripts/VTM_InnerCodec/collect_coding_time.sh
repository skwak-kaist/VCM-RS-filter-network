#!/bin/bash

eval "$(conda shell.bash hook)"

env_name=$(head -n 1 ../../vcm_env_name.txt | xargs)
echo $env_name
conda activate $env_name

root=output

datasets="SFU TVD_tracking"

encoding_modes="AI_e2e AI_inner LD_e2e LD_inner RA_e2e RA_inner"

for dataset in $datasets; do

	for encoding_mode in $encoding_modes; do

		#python collect_coding_time.py $root/${dataset}_${encoding_mode}/coding_log encoding
		echo "encoding time computed"
		
	done
done


for dataset in $datasets; do

	for encoding_mode in $encoding_modes; do

		python collect_coding_time.py $root/decode/${dataset}_${encoding_mode}/decoding_log decoding
		
	done
done
