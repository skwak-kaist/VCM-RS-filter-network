#!/usr/bin/env python

# process SUF-HW dataset

import os
import sys
from pathlib import Path
import re
import numpy as np
import time
import shutil
import sys
import vcmrs
import pandas as pd

print(os.path.abspath(__file__))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sfu_config
import utils

base_folder = "output"
timestamp = time.strftime('%Y%m%d_%H%M%S')

scenario = sys.argv[1]
test_id = f'SFU_{scenario}_{timestamp}'

if 'RA' in scenario: gop_configuration = 'RandomAccess'
if 'LD' in scenario: gop_configuration = 'LowDelay'
if 'AI' in scenario: gop_configuration = 'AllIntra'

output_dir = f'{base_folder}/{test_id}'
log_dir = f"{output_dir}/coding_log"
os.makedirs(log_dir, exist_ok=True)
vcmrs.setup_logger(name = "coding", logfile = f"{log_dir}/main.log")

##############################################################
# configuration
cuda_devices = [0, 1, 2, 3, 4, 5, 6, 7]
processes_per_gpu = 1 # number of process running per GPU
# for 24G GPU, 2 processes per GPU may be used
# each GPU process for a class
cpu_per_gpu_process = 5 if 'AI' in scenario else 10 # number of cpu process for each GPU process

# classes to be evaluated
sfu_classes = ['A', 'B', 'C', 'D', 'O']

data_dir = '../../Data/SFU'

utils.check_descriptor_files(data_dir)

#vcmrs.log=print
# set test_id and output_dir
vcmrs.log(f'Test ID: {test_id}')
vcmrs.log(f'Output directory: {output_dir}\n')

#####################################################################
# prepare tasks
tasks = []
log_files = []
for seq_id in sfu_config.seq_cfg[scenario].keys():
  if not (seq_id in sfu_config.seq_dict.keys()): continue

  seq_class, fname = sfu_config.seq_dict[seq_id]
  if seq_class not in sfu_classes: continue
  for qp_idx, enc_cfg in enumerate(sfu_config.seq_cfg[scenario][seq_id]):
    # enc_cfg has format (quality, NNIntraQPOffset)
    quality, nn_intra_qp_offset = enc_cfg
    qp = f'qp{qp_idx}'

    bitstream_fname = os.path.join('bitstream', fname, f'{fname}_{qp}.bin')
    recon_fname = os.path.join('recon', f'{fname}_{qp}')

    # process chunks in parallel
    working_dir = os.path.join(base_folder, 'working_dir', test_id, f'{seq_id}_{qp}')
    log_file = f"{log_dir}/encoding_{seq_id}_{qp}_{timestamp}.log"
    cfg_file = f"{log_dir}/encoding_{seq_id}_{qp}_{timestamp}.cfg"

    video_fname = os.path.join(data_dir, f'Class{seq_class}', f'{fname}.yuv')

    cfg = {}
    
    common_ini_file = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"common.ini" )
    utils.update_cfg_from_ini( common_ini_file, cfg)

    machine_task_ini_file = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"sfu.ini" )
    utils.update_cfg_from_ini( machine_task_ini_file, cfg)

    cfg["SourceWidth"] = sfu_config.res_dict[seq_class][0]
    cfg["SourceHeight"] = sfu_config.res_dict[seq_class][1]
    cfg["FrameRate"] = sfu_config.fr_dict[seq_id][1]
    cfg["IntraPeriod"] = sfu_config.fr_dict[seq_id][0]
    cfg["quality"] = quality
    cfg["NNIntraQPOffset"] = nn_intra_qp_offset
    cfg["working_dir"] = working_dir
    cfg["output_dir"] = output_dir
    cfg["output_bitstream_fname"] = bitstream_fname
    cfg["output_recon_fname"] = recon_fname
    cfg["logfile"] = log_file
    cfg["FramesToBeEncoded"] = sfu_config.fr_dict[seq_id][2]
    cfg["FrameSkip"] = sfu_config.fr_dict[seq_id][3]
    cfg["InputBitDepth"] = 8
    cfg["InputChromaFormat"] = 420
    cfg["InnerCodec"] = 'NNVVC'
    cfg["num_workers"] = cpu_per_gpu_process
    cfg["RoIGenerationNetwork"] = sfu_config.seq_roi_cfg_network

    scenario_ini_file = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"{scenario}.ini" )
    utils.update_cfg_from_ini( scenario_ini_file, cfg)

    utils.set_descriptor_files(data_dir, scenario, cfg, 'SFU', fname)

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

    vcmrs.log('\n'.join(map(str,cmd)))
    with open(cfg_file, 'w', newline='\n') as f:
      f.write('\n'.join(map(str,cmd)))
    tasks.append(cmd)
    log_files.append(log_file)

###############################################
# encoding
start_time = time.time()
rpool = utils.GPUPool(device_ids = cuda_devices, proc_per_dev = processes_per_gpu)
err = rpool.process_tasks(tasks)
vcmrs.log('\n\n=================================================')
vcmrs.log(f'Encoding error code: {err}')
vcmrs.log(f'Encoding elapse: {time.time() - start_time}, GPUs {len(cuda_devices)}')
vcmrs.log('\n\n')

#### Collect coding times #######
times = utils.collect_coding_times(log_files, seq_order_list = list(sfu_config.seq_dict.keys()), in_hours=True)
vcmrs.log("Coding time by QP:")
vcmrs.log(times)

