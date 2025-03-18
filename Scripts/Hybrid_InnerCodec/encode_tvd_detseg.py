#!/usr/bin/env python

# Encode TVD image dataset

import os
import sys
import numpy as np
import time
import shutil
import vcmrs

print(os.path.abspath(__file__))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

timestamp = time.strftime('%Y%m%d_%H%M%S')
test_id = f"TVD_detseg_{timestamp}"
base_folder = "output"
output_dir = f'{base_folder}/{test_id}'
working_dir = f'{base_folder}/working_dir/{test_id}'
log_dir =  f"{output_dir}/coding_log"
vcmrs.setup_logger(name = "coding", logfile = f"{log_dir}/main.log")
##############################################################
qps = [22, 27, 32, 37, 42, 47]
cuda_devices = [0,1,2,3,4,5,6,7] # GPU devices
processes_per_gpu = 1 # 16G memeory


vcmrs.log('processing TVD_detseg...')
vcmrs.log(f'Test ID: {test_id}')

data_dir = '../../Data/tvd_object_detection_dataset'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

################################################################

tasks = []
task_name = []
for quality in qps:
  bitstream_fname = os.path.join('bitstream', f'QP_{quality}', '{bname}.bin')
  recon_fname = os.path.join('recon', f'QP_{quality}', '{bname}')
  log_file = f"{log_dir}/qp{quality}.log"
  cfg_file =  f"{log_dir}/qp{quality}.cfg"
  working_dir = os.path.join(working_dir, f"QP{quality}")

  cfg = {}
  cfg["quality"] = quality
  cfg["working_dir"] = working_dir
  cfg["output_dir"] = output_dir
  cfg["output_bitstream_fname"] = bitstream_fname
  cfg["output_recon_fname"] = recon_fname
  cfg["logfile"] = log_file
  cfg["InnerCodec"] = 'NNVVC'
  #cfg["num_workers"] = num_workers

  for i in range(1, len(sys.argv)):
    if sys.argv[i].startswith("--"):
      key = sys.argv[i][2:]
      if i+1<len(sys.argv) and not sys.argv[i+1].startswith("--"):
        cfg[key] = sys.argv[i+1]
      else:
        cfg[key] = None

  cmd = ['python',
    '-m', 'vcmrs.encoder',
    '--debug_source_checksum'
    ]

  for c in cfg.keys():
    cmd.append('--'+c)
    if cfg[c] is not None: cmd.append(str(cfg[c]))

  cmd.append(data_dir)

  with open(cfg_file, 'w', newline='\n') as f:
    f.write('\n'.join(map(str,cmd)))

  tasks.append(cmd)
  task_name.append(f"QP{quality}")

start_time = time.time()
rpool = utils.GPUPool(device_ids = cuda_devices, proc_per_dev = processes_per_gpu)
err = rpool.process_tasks(tasks)
err_dict = {k:v for k,v in zip(task_name, err)}
vcmrs.log(f'error code: {err_dict}')
vcmrs.log(f'total elapse: {time.time() - start_time}')
vcmrs.log('all process has finished')

### Collect coding time ####
times = utils.collect_coding_times([f"{log_dir}/qp{qp}.log" for qp in qps], file_regx="(.*?)(qp\d+).*")
vcmrs.log("Coding time by QP:")
vcmrs.log(times)
