#!/bin/bash

eval "$(conda shell.bash hook)"

env_name=$(head -n 1 ../../vcm_env_name.txt | xargs)
echo $env_name
conda activate $env_name


filter_mode=rgb_post_vtm_multi_qp_finetuned
decode_folder=decode_postfilter_vtm_bdr_off_colorize_luma_pre_shift_off

dataset=SFU # SFU, TVD_tracking
scenario=$1

run_decode=1
run_eval=0

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
if [ $run_decode -eq 1 ]; then
    if [ $dataset == "SFU" ]; then
        ./decode_sfu_postfilter.sh SFU_${scenario_name} ${filter_mode} ${decode_folder} ${gpu_id}
    elif [ $dataset == "TVD_tracking" ]; then
        ./decode_tvd_tracking_postfilter.sh TVD_tracking_${scenario_name} ${filter_mode} ${decode_folder} ${gpu_id}
    else
        echo "Invalid dataset"
        exit
    fi
fi


if [ $run_eval -eq 1 ]; then
    # run evaluation
    cd /home/skwak/Workspace/Project_VCM/vcm-ctc/eval_scripts

    # dataset이 SFU이면
    if [ $dataset == "SFU" ]; then
        ./eval_SFU_postfilter.sh $scenario_name $decode_folder $gpu_id
    # dataset이 TVD_tracking이면
    elif [ $dataset == "TVD_tracking" ]; then
        ./eval_TVD_postfilter.sh $scenario_name $decode_folder $gpu_id
    else
        echo "Invalid dataset"
        exit
    fi
fi


