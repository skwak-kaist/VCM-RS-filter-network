#!/usr/bin/env python

# process open images

import os
import sys
import shutil
import numpy as np
import glob
import time
import vcmrs

print(os.path.abspath(__file__))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

timestamp = time.strftime('%Y%m%d_%H%M%S')
test_id = f'openimages_{timestamp}'
base_folder = "output"
output_dir = f'{base_folder}/{test_id}'
working_dir = f'{base_folder}/working_dir/{test_id}'
log_dir = f"{output_dir}/coding_log"
vcmrs.setup_logger(name = "coding", logfile = f"{log_dir}/main.log")
##############################################################
qps = [22, 27, 32, 37, 42, 47]
cuda_devices = [0, 1, 2, 3, 4, 5, 6, 7] # CUDA devices
processes_per_gpu = 1 # number of processes running per GPU
n_chunks = 6 # split images into chunks, so they can be parallelized better

vcmrs.log('Encoding OpenImages...')
vcmrs.log(f'Test ID: {test_id}')

data_dir = '../../Data/OpenImages/validation/'

# prepare working dir
os.makedirs(working_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

################################################################
# get input images
input_fnames = sorted(glob.glob(os.path.join(data_dir, '*.jpg'))) + sorted(glob.glob(os.path.join(data_dir, '*.png')))
input_fnames = [ os.path.basename(path) for path in input_fnames]

#with open('openimages_input.lst', 'r') as f:
#  input_fnames = f.read().splitlines()

# split input images into chuncks
chunks = np.array_split(input_fnames, n_chunks)
chunk_fnames = []
for idx in range(n_chunks):
  chunk_fname = os.path.join(working_dir, f'openimages_{idx}.ini')
  with open(chunk_fname, 'w') as of:
      of.write('['+' '.join(chunks[idx])+']')
      of.write(f'\ninput_prefix = {data_dir}')
  chunk_fnames.append(chunk_fname)

# prepare tasks
tasks = []
task_name = []
for quality in qps:
  bitstream_fname = os.path.join('bitstream', f'QP_{quality}', '{bname}.bin')
  recon_fname = os.path.join('recon', f'QP_{quality}', '{bname}')

  for chunk_idx,fname in enumerate(chunk_fnames):

    log_file = f"{log_dir}/qp{quality}_{chunk_idx}.log"
    cfg_file = f"{log_dir}/qp{quality}_{chunk_idx}.cfg"
    working_dir = os.path.join(working_dir, f"QP{quality}_{chunk_idx}")

    cfg = {}
    cfg["quality"] = quality
    cfg["working_dir"] = working_dir
    cfg["output_dir"] = output_dir
    cfg["output_bitstream_fname"] = bitstream_fname
    cfg["output_recon_fname"] = recon_fname
    cfg["logfile"] = log_file
    cfg["InnerCodec"] = 'NNVVC'
    #cfg["num_workers"] = num_workers

    cmd = ['python',
      '-m', 'vcmrs.encoder',
      '--debug_source_checksum'
      ]

    for c in cfg.keys():
      cmd.append('--'+c)
      if cfg[c] is not None: cmd.append(str(cfg[c]))

    cmd.append(fname)

    with open(cfg_file, 'w', newline='\n') as f:
      f.write('\n'.join(map(str,cmd)))

    tasks.append(cmd)
    task_name.append(f"QP{quality}_{chunk_idx}")

start_time = time.time()
rpool = utils.GPUPool(device_ids = cuda_devices, proc_per_dev = processes_per_gpu)
err = rpool.process_tasks(tasks)
err_dict = {k:v for k,v in zip(task_name, err)}
vcmrs.log(f'error code: {err_dict}')
vcmrs.log(f'total elapse: {time.time() - start_time}')
vcmrs.log('all process has finished')

### Collect coding time ####
times = utils.collect_coding_times([f"{log_dir}/qp{qp}.log" for qp in qps], file_regx="(.*?)(qp\d+).*")
# times['encoding_time']=times['encoding_time']/8189*5000
vcmrs.log("Coding time by QP on 5000 images:")
vcmrs.log(times)
times.to_csv(os.path.join(log_dir,"codingtime_estimated.csv"))
