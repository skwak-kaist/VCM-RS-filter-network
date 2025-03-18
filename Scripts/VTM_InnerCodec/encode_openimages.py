#!/usr/bin/env python

# Encode TVD image dataset using VTM as the inner codec
# totoal number of tasks 49134
#
# Usage of the script: 
#   <script_name> <task_id>
# This script may be useful to process the data on a cluster using CPUs. 
# The script process one item identified by the task_id. The task id is from
# 1 to the total number of tasks. 
# 


import os
import sys
import glob
import time
import subprocess

print(os.path.abspath(__file__))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

timestamp = time.strftime('%Y%m%d_%H%M%S')
test_id = f"OpenImages"
output_dir = f'./output/{test_id}'
log_dir=f"{output_dir}/coding_log"
print(f'processing {test_id}...')
##############################################################
qps = [22, 27, 32, 37, 42, 47]

# use only CPU
num_workers = 1 # This number should match the number of cores a node has

data_dir = '../../Data/OpenImages/validation'
working_dir = f'./output/working_dir/{test_id}'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

################################################################
# get all images
img_fnames = sorted(glob.glob(os.path.join(data_dir, '*.jpg'))) + sorted(glob.glob(os.path.join(data_dir, '*.png')))
#with open('../openimages_input.lst', 'r') as f:
#  img_fnames = f.read().splitlines()
#img_fnames=[os.path.join(data_dir, x) for x in img_fnames]

total_tasks = len(img_fnames) * len(qps)
# total number of tasks 996
print('Total number of tasks: ', total_tasks)

#get task id from input arguments
task_id = int(sys.argv[1]) - 1

quality = qps[task_id // len(img_fnames)]
img_fname = img_fnames[task_id % len(img_fnames)]

bitstream_fname = os.path.join('bitstream', f'QP_{quality}', '{bname}.bin')
recon_fname = os.path.join('recon', f'QP_{quality}', '{bname}')
log_file = f"{log_dir}/qp{quality}_{task_id}.log"
cfg_file = f"{log_dir}/qp{quality}_{task_id}.cfg"
working_dir = os.path.join(working_dir, str(quality))

cfg = {}
cfg["quality"] = quality
cfg["working_dir"] = working_dir
cfg["output_dir"] = output_dir
cfg["output_bitstream_fname"] = bitstream_fname
cfg["output_recon_fname"] = recon_fname
cfg["logfile"] = log_file
cfg["InnerCodec"] = 'VTM'
cfg["num_workers"] = num_workers

for i in range(2, len(sys.argv)):
  if sys.argv[i].startswith("--"):
    key = sys.argv[i][2:]
    if i+1<len(sys.argv) and not sys.argv[i+1].startswith("--"):
      cfg[key] = sys.argv[i+1]
    else:
      cfg[key] = None

cmd = ['python', 
      '-m', 'vcmrs.encoder',
      '--debug',
      '--debug_source_checksum'
      ]

for c in cfg.keys():
  cmd.append('--'+c)
  if cfg[c] is not None: cmd.append(str(cfg[c]))

cmd.append(img_fname)

print('\n'.join(map(str,cmd)))
with open(cfg_file, 'w', newline='\n') as f:
  f.write('\n'.join(map(str,cmd)))

start_time = time.time()
subprocess.run(map(str, cmd))
print(f'Elapse: {time.time()-start_time}')
print('all done')


