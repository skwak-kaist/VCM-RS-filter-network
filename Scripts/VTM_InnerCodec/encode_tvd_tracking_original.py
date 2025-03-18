#!/usr/bin/env python

# process TVD dataset using VTM as the inner codec
# number of tasks 42
#
# Usage of the script: 
#   <script_name> <task_id> <scenario>
# This script may be useful to process the data on a cluster using CPUs. 
# The script process one item identified by the task_id. The task id is from
# 1 to the total number of tasks. 
# <scenario> is one of: the following:
# RA, LD, AI - old scenarios
# RA_inner, LD_inner, AI_inner - current inner scenarios
# RA_e2e, LD_e2e, AI_e2e - current end-to-end scenarios


import os
import sys
import numpy as np
import time
import shutil

print(os.path.abspath(__file__))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tvd_tracking_config as config
import utils

timestamp = time.strftime('%Y%m%d_%H%M%S')

# set test_id and output_dir
scenario = sys.argv[2]
test_id = f'TVD_tracking_{scenario}'
output_dir = f'./output/{test_id}'
log_dir = f"{output_dir}/coding_log"
os.makedirs(log_dir, exist_ok=True)

print(f'Test ID: {test_id}')
print(f'Output directory: {output_dir}\n')

##############################################################
# configuration
num_workers = 1 # for CPU encoding, one CPU per video. This number should match the number of cores a node has

#data_dir = '../../Data/TVD'
data_dir = '../Data/TVD'

utils.check_descriptor_files(data_dir)

################################################################
# prepare environment

# set test_id
task_id = int(sys.argv[1]) - 1

# get number of tasks
seqs = list(config.seq_cfg[scenario].keys())
num_seqs = len(seqs)
num_tasks = num_seqs * len(config.seq_cfg[scenario]['TVD-01_1'])

seq_idx = task_id % num_seqs
qp_idx = task_id // num_seqs
print(f'Total number of task: {num_tasks}, task id: {task_id}, seq_id: {seq_idx}, qp_idx: {qp_idx}')

video_id = seqs[seq_idx]
quality, nn_intra_qp_offset = config.seq_cfg[scenario][video_id][qp_idx]
base_video = video_id.split('_')[0]

print(f'processing seq {base_video} with {qp_idx}: {quality}')

video_fname = os.path.join(data_dir, f"{base_video}.yuv")

bitstream_fname = os.path.join('bitstream', f'{video_id}_qp{qp_idx}.bin')
recon_fname = os.path.join('recon', f'qp{qp_idx}', f'{video_id}')
working_dir = f"./output/working_dir/{test_id}/{video_id}_qp{qp_idx}"
log_file = f"{log_dir}/encoding_{video_id}_qp{qp_idx}_{timestamp}.log"
cfg_file = f"{log_dir}/encoding_{video_id}_qp{qp_idx}_{timestamp}.cfg"

cfg = {}

common_ini_file = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"common.ini" )
utils.update_cfg_from_ini( common_ini_file, cfg)

machine_task_ini_file = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"tvd_tracking.ini" )
utils.update_cfg_from_ini( machine_task_ini_file, cfg)

cfg["SourceWidth"] = 1920
cfg["SourceHeight"] = 1080
cfg["FrameRate"] = 50
cfg["IntraPeriod"] = 64
cfg["quality"] = quality
cfg["NNIntraQPOffset"] = nn_intra_qp_offset
cfg["working_dir"] = working_dir
cfg["output_dir"] = output_dir
cfg["output_bitstream_fname"] = bitstream_fname
cfg["output_recon_fname"] = recon_fname
cfg["logfile"] = log_file
cfg["FramesToBeEncoded"] = config.fr_dict[video_id][2]
cfg["FrameSkip"] = config.fr_dict[video_id][3]
cfg["InputBitDepth"] = config.fr_dict[video_id][4]
cfg["InputChromaFormat"] = 420
cfg["InnerCodec"] = 'VTM'
cfg["num_workers"] = num_workers
cfg["RoIGenerationNetwork"] = config.seq_roi_cfg_network

scenario_ini_file = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"{scenario}.ini" )
utils.update_cfg_from_ini( scenario_ini_file, cfg)

utils.set_descriptor_files(data_dir, scenario, cfg, 'TVD', video_id)

for i in range(3, len(sys.argv)):
  if sys.argv[i].startswith("--"):
    key = sys.argv[i][2:]
    if i+1<len(sys.argv) and not sys.argv[i+1].startswith("--"):
      cfg[key] = sys.argv[i+1]
    else:
      cfg[key] = None

################################################  

cmd = ['python', 
      '-m', 'vcmrs.encoder',
      '--directory_as_video',
      ]

for c in cfg.keys():
  cmd.append('--'+c)
  if cfg[c] is not None: cmd.append(str(cfg[c]))

cmd.append(video_fname)

print('\n'.join(map(str,cmd)))
with open(cfg_file, 'w', newline='\n') as f:
  f.write('\n'.join(map(str,cmd)))

################################################  
# encoding
start_time = time.time()
os.system(' '.join(map(str,cmd)))
end_time = time.time()
print('\n\n=================================================')
print(f'Encoding elapse: {end_time-start_time}')
print('all done\n\n')
