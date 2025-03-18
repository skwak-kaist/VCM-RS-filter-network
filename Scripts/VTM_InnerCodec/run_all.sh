#!/bin/bash

eval "$(conda shell.bash hook)"

env_name=$(head -n 1 ../../vcm_env_name.txt | xargs)
echo $env_name
conda activate $env_name

set -ex

echo '''
  encoding 
'''
num_workers=50
total_num_GPUs=2

echo "Start encoding all using $num_workers CPU cores"

# SFU
num_tasks=84  
for task_id in $(seq $num_tasks); do
  gpu_id=$((($task_id - 1) % $total_num_GPUs)) 
  echo $task_id
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_sfu_AI_e2e.py $task_id 
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_sfu_RA_e2e.py $task_id
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_sfu_LD_e2e.py $task_id
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_sfu_AI_inner.py $task_id 
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_sfu_RA_inner.py $task_id
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_sfu_LD_inner.py $task_id
done

# TVD tracking
num_tasks=42
for task_id in $(seq $num_tasks); do
  gpu_id=$((($task_id - 1) % $total_num_GPUs)) 
  echo $gpu_id
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_tvd_tracking_AI_e2e.py $task_id 
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_tvd_tracking_RA_e2e.py $task_id 
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_tvd_tracking_LD_e2e.py $task_id 
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_tvd_tracking_AI_inner.py $task_id 
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_tvd_tracking_RA_inner.py $task_id 
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_tvd_tracking_LD_inner.py $task_id 
done

# Pandaset
if false; then
# num_tasks=438 # for class 1-6
num_tasks=216 # for class 1-3
for task_id in $(seq $num_tasks); do
  gpu_id=$((($task_id - 1) % $total_num_GPUs)) 
  echo $gpu_id
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_pandaset.py --scenario RA_e2e --task_id $task_id 
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_pandaset.py --scenario LD_e2e--task_id $task_id 
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_pandaset.py --scenario AI_e2e --task_id $task_id 
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_pandaset.py --scenario RA_inner --task_id $task_id 
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_pandaset.py --scenario LD_inner --task_id $task_id 
  sem -j $num_workers CUDA_VISIBLE_DEVICES=$gpu_id python encode_pandaset.py --scenario AI_inner --task_id $task_id 
done
fi

sem  --wait

echo "Start encoding all using $num_workers CPU cores"
