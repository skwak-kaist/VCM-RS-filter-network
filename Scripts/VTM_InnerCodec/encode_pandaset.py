#!/usr/bin/env python

# process SUF-HW dataset 
# number of tasks 102

import os
import sys
import numpy as np
import time
import shutil
import filecmp
import atexit
import tempfile
from types import SimpleNamespace

print(os.path.abspath(__file__))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandaset_config
import utils

##############################################################
# intput arguments

known_args = {}
known_args["test_id"] = None        # Test ID, Default: output.
known_args["scenario"] = "RA_inner" # scenario, RA/LD/AI + _inner/_e2e/_old, Default: RA_inner
known_args["seq_id"] = None         # sequence id, default: None
known_args["seq_qp"] = None         # sequence id, default: None
known_args["classes"] = "1,2,3"     # classes, default: "1,2,3"
known_args["task_id"] = 1           # Task ID, starting from 1, Default: 0.

for i in range(1, len(sys.argv)):
  if sys.argv[i].startswith("--"):
    key = sys.argv[i][2:]
    if key not in known_args: continue
    if i+1<len(sys.argv) and not sys.argv[i+1].startswith("--"):
      try:
        known_args[key] = int(sys.argv[i+1])
      except:
        known_args[key] = sys.argv[i+1]
    else:
      known_args[key] = None

args = SimpleNamespace(**known_args)

assert args.task_id>=1, 'task_id shall start from 1'

num_workers = 1
scenario = args.scenario

timestamp = time.strftime('%Y%m%d_%H%M%S')
log_dir = f"coding_log/pandaset_{timestamp}"
os.makedirs(log_dir, exist_ok=True)


##############################################################
# configuration
data_dir = '../../Data/Pandaset_YUV'

utils.check_descriptor_files(data_dir)

# set test_id and output_dir
test_id = args.test_id
if test_id is None: test_id = f'Pandaset_{args.scenario}'

output_dir = f'./output/{test_id}'
print(f'Test ID: {test_id}')
print(f'Output directory: {output_dir}\n')

##############################################################
# CTC config

tasks = []
if args.seq_id is None: 
  # get all seq_ids for the class to be processed
  seq_ids_in_classes = []
  for cls in args.classes.split(','):
    seq_ids_in_classes += pandaset_config.cls_dict[str(cls)]

  # encode whole dataset
  for seq_id, (dataset, seq_name) in pandaset_config.seq_dict.items():
    if not int(seq_id) in seq_ids_in_classes: continue 
    for qp_idx,(quality, nn_intra_qp_offset) in enumerate(pandaset_config.seq_cfg[scenario][seq_id]):
      tasks.append((seq_id, qp_idx, quality, scenario))
else:
  seq_id = args.seq_id
  tasks.append((seq_id, -1, args.seq_qp, scenario))
  

###############################################################

print(f'Total number of task: {len(tasks)}')

# get number of tasks
task_id = args.task_id - 1
seq_id, qp_idx, quality, scenario = tasks[task_id]

print('Tasks: ')
print(*list(zip(range(len(tasks)), tasks)), sep='\n')

print(f'Encoding seq: {seq_id}, qp_idx: {qp_idx}, quality: {quality}, scenario: {scenario}')

seq_class,fname = pandaset_config.seq_dict[seq_id]
if qp_idx == -1:
  bitstream_fname = os.path.join('bitstream', f'{fname}_{quality}.bin')
  recon_fname = os.path.join('recon', f'{fname}_{quality}')
else:
  bitstream_fname = os.path.join('bitstream', f'{fname}_qp{qp_idx}.bin')
  recon_fname = os.path.join('recon', f'{fname}_qp{qp_idx}')

video_fname = os.path.join(data_dir, f'{fname}.yuv')
working_dir = os.path.join('output', 'working_dir', test_id, f'{seq_id}_{quality}')
log_file = f"{log_dir}/pandaset_{seq_id}_{quality}.log"
cfg_file = f"{log_dir}/pandaset_{seq_id}_{quality}.cfg"

# get seqence information
#cls = pandaset_config.seq_dict[seq_id][0]
cfg = {}

common_ini_file = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"common.ini" )
utils.update_cfg_from_ini( common_ini_file, cfg)

machine_task_ini_file = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"pandaset.ini" )
utils.update_cfg_from_ini( machine_task_ini_file, cfg)

cfg["SourceWidth"] = 1920
cfg["SourceHeight"] = 1080
cfg["FrameRate"] = pandaset_config.fr_dict[seq_id][1]
cfg["IntraPeriod"] = pandaset_config.fr_dict[seq_id][0]
cfg["quality"] = quality
cfg["NNIntraQPOffset"] = nn_intra_qp_offset
cfg["working_dir"] = working_dir
cfg["output_dir"] = output_dir
cfg["output_bitstream_fname"] = bitstream_fname
cfg["output_recon_fname"] = recon_fname
cfg["logfile"] = log_file
cfg["FramesToBeEncoded"] = pandaset_config.fr_dict[seq_id][2]
cfg["FrameSkip"] = pandaset_config.fr_dict[seq_id][3]
cfg["InputBitDepth"] = 8
cfg["InputChromaFormat"] = 420
cfg["InnerCodec"] = 'VTM'
cfg["num_workers"] = num_workers
cfg["RoIGenerationNetwork"] = pandaset_config.seq_roi_cfg_network

scenario_ini_file = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"{scenario}.ini" )
utils.update_cfg_from_ini( scenario_ini_file, cfg)

utils.set_descriptor_files(data_dir, scenario, cfg, 'Pandaset', fname)

for i in range(1, len(sys.argv)):
  if sys.argv[i].startswith("--"):
    key = sys.argv[i][2:]
    if key in known_args: continue
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
os.system(' '.join(map(str, cmd)))

print('\n\n=================================================')
print(f'Encoding elapse: {time.time()-start_time}')
print('\n\n')
