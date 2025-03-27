#!/bin/bash

filter_mode=rgb_post_lic_multi_quality
decode_folder=decode_postfilter_lic_bdr_off

dataset=TVD_tracking # SFU, TVD_tracking
scenario=$1

# scenario가 1이면 AI_e2e, 2이면 AI_inner, 3이면 LD_e2e, 4이면 LD_inner, 5이면 RA_e2e, 6이면 RA_inner
if [ $scenario -eq 1 ]; then
    scenario_name=AI_e2e
    gpu_id=0
elif [ $scenario -eq 2 ]; then
    scenario_name=AI_inner
    gpu_id=1
elif [ $scenario -eq 3 ]; then
    scenario_name=LD_e2e
    gpu_id=0
elif [ $scenario -eq 4 ]; then
    scenario_name=LD_inner
    gpu_id=1
elif [ $scenario -eq 5 ]; then
    scenario_name=RA_e2e
    gpu_id=0
elif [ $scenario -eq 6 ]; then
    scenario_name=RA_inner
    gpu_id=1
else
    echo "Invalid scenario"
    exit
fi




# run decoding

# dataset이 SFU이면
if [ $dataset == "SFU" ]; then
    ./decode_sfu_postfilter.sh SFU_${scenario_name} ${filter_mode} ${decode_folder} ${gpu_id}
# dataset이 TVD_tracking이면
elif [ $dataset == "TVD_tracking" ]; then
    ./decode_tvd_tracking_postfilter.sh TVD_tracking_${scenario_name} ${filter_mode} ${decode_folder} ${gpu_id}
else
    echo "Invalid dataset"
    exit
fi

# run evaluation

cd /home/skwak/Workspace/Project_VCM/vcm-ctc/eval_scripts

./eval_TVD_postfilter.sh $scenario_name $decode_folder $gpu_id
