#!/bin/bash

# Decode TVD tracking bitstreams

eval "$(conda shell.bash hook)"

env_name=$(head -n 1 ../../vcm_env_name.txt | xargs)
echo $env_name
conda activate $env_name



set -ex

#test_id='TVD_tracking'
test_id=$1
filter_mode=$2
decode_folder=$3
cuda_device=$4
base_folder="output"
input_dir="$base_folder/$test_id/bitstream"
output_dir="$base_folder/${decode_folder}/$test_id/recon"
log_dir="$base_folder/${decode_folder}/$test_id/decoding_log"
qps='qp0 qp1 qp2 qp3 qp4 qp5'

mkdir -p $output_dir

video_ids='TVD-01_1 TVD-01_2 TVD-01_3 TVD-02_1 TVD-03_1 TVD-03_2 TVD-03_3'

#
start_time=$(date +%s)
for video_id in $video_ids; do
  for qp in $qps; do
    bs_fname=$input_dir/${video_id}_${qp}.bin
    echo processing $bs_fname ...
    if test -f "$bs_fname"; then
      CUDA_VISIBLE_DEVICES=$cuda_device python -m vcmrs.decoder \
        --InnerCodec VTM \
        --working_dir "$base_folder/dec_temp/$test_id" \
        --output_dir $output_dir/$qp \
        --output_recon_fname $video_id \
        --debug_source_checksum \
        --logfile "${log_dir}/${video_id}_${qp}.log" \
        --JointFilterPostModel 'FilterV8' \
        --JointFilterSelectionAlgorithm ${filter_mode} \
        $bs_fname
    fi
  done
done

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo 
echo  decoding time: $runtime seconds
echo 
echo  Decoding completed


