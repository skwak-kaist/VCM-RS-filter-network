#!/usr/bin/env python

# process SUF-HW dataset using VTM as the inner codec
# number of tasks 84
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
from pathlib import Path

print(os.path.abspath(__file__))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sfu_config
import utils

timestamp = time.strftime('%Y%m%d_%H%M%S')

scenario = sys.argv[2]
test_id = f'SFU_{scenario}'
output_dir = Path(sys.argv[3]) / test_id
log_dir = f"{output_dir}/coding_log"
os.makedirs(log_dir, exist_ok=True)

print(f'Test ID: {test_id}')
print(f'Output directory: {output_dir}\n')

##############################################################
# configuration
num_workers = 1 # for CPU encoding, one CPU per video. This number should match the number of cores a node has

data_dir = '../Data/SFU'

utils.check_descriptor_files(data_dir)

################################################################
# prepare environment

# set test_id
task_id = int(sys.argv[1]) - 1

# get number of tasks
seq_ids = list(sfu_config.seq_cfg[scenario].keys())
num_seqs = len(seq_ids)
num_tasks = num_seqs * len(sfu_config.seq_cfg[scenario][seq_ids[0]])

seq_idx = task_id % num_seqs
qp_idx = task_id // num_seqs
print(f'Total number of task: {num_tasks}, task id: {task_id}, seq_id: {seq_idx}, qp_idx: {qp_idx}')

seq_id = list(sfu_config.seq_dict.keys())[seq_idx]
qp = f'qp{qp_idx}'
seq_class,fname = sfu_config.seq_dict[seq_id]
quality, nn_intra_qp_offset = sfu_config.seq_cfg[scenario][seq_id][qp_idx]

print(f'processing seq {fname} with {qp}: {quality}')

video_fname = os.path.join(data_dir, f'Class{seq_class}', f'{fname}.yuv')

bitstream_fname = os.path.join('bitstream', fname, f'{fname}_{qp}.bin')
recon_fname = os.path.join('recon', f'{fname}_{qp}')
working_dir = Path(sys.argv[3]) / 'working_dir' / test_id / f'{seq_id}_{qp}'
log_file = f"{log_dir}/encoding_{seq_id}_{qp}_{timestamp}.log"
cfg_file = f"{log_dir}/encoding_{seq_id}_{qp}_{timestamp}.cfg"

cfg = {}

common_ini_file = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"common.ini" )
utils.update_cfg_from_ini( common_ini_file, cfg)

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
cfg["InnerCodec"] = 'VTM'
cfg["num_workers"] = num_workers
cfg["RoIGenerationNetwork"] = sfu_config.seq_roi_cfg_network

# Modification for Joint Filter Module
component_table = {
  'F': 'FormatAdapter',
  'T': 'TemporalResample',
  'S': 'SpatialResample',
  'R': 'ROI',
  'B': 'BitDepthTruncation',
  'J': 'JointFilter',
  'P': 'PostFilter',
}
component_order = sys.argv[4]
pre_components_str, post_components_str = component_order.split('_')
pre_components_str = 'F' + pre_components_str
post_components_str = post_components_str + 'F'

cfg['ComponentOrder'] = component_order
for k, v in component_table.items():
  if (k not in ['J', 'P']) and k not in pre_components_str:
    assert k not in post_components_str
    cfg[v] = 'Bypass'
cfg['JointFilterSelectionAlgorithm'] = sys.argv[5]

cfg['JointFilterPreDomain'] = sys.argv[6]
cfg['JointFilterPostDomain'] = sys.argv[7]
cfg['JointFilterPreModel'] = sys.argv[8]
cfg['JointFilterPostModel'] = sys.argv[9]

cfg['BitTruncationRestorationWidthThreshold'] = int(sys.argv[10])
cfg['BitTruncationRestorationHeightThreshold'] = int(sys.argv[11])

scenario_ini_file = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"{scenario}.ini" )
utils.update_cfg_from_ini( scenario_ini_file, cfg)

#roi_descriptor, spatial_descriptor = utils.get_descriptor_files(data_dir, scenario, cfg, 'SFU', fname)
#cfg["RoIDescriptor"] = roi_descriptor
#cfg["SpatialDescriptor"] = spatial_descriptor

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
