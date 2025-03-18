#!/usr/bin/env python

# process TVD dataset

import os
import sys
import numpy as np
import time
import shutil
import vcmrs

print(os.path.abspath(__file__))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tvd_tracking_config as config
import utils

timestamp = time.strftime('%Y%m%d_%H%M%S')

scenario = sys.argv[1]
test_id = f'TVD_tracking_{scenario}_{timestamp}'
base_folder = "output"
output_dir = f'{base_folder}/{test_id}'
log_dir = f"{output_dir}/coding_log"
os.makedirs(log_dir, exist_ok=True)
vcmrs.setup_logger(name="coding", logfile=f"{log_dir}/main.log")
##############################################################

cuda_devices = [0,1,2,3,4,5,6,7]
processes_per_gpu = 2 # number of process running per GPU
cpu_per_gpu_process = 10 # number of cpu process for each GPU process

data_dir = '../../Data/TVD'

utils.check_descriptor_files(data_dir)

# check input video md5sum
#print('checking input video md5sum...')
#for video_id, md5sum in config.md5sum.items():
#  utils.check_md5sum(os.path.join(data_dir, f"{video_id}.yuv"), md5sum)

################################################################
# encoding

tasks = []
log_files = []
start_time = time.time()
for video_id in config.seq_cfg[scenario].keys():
  base_video = video_id.split('_')[0]
  video_fname = os.path.join(data_dir, f"{base_video}.yuv")
  for qp_idx,(quality, nn_intra_qp_offset) in enumerate(config.seq_cfg[scenario][video_id]):
    bitstream_fname = os.path.join('bitstream', f'{video_id}_qp{qp_idx}.bin')
    recon_fname = os.path.join('recon', f'qp{qp_idx}', f'{video_id}')
    output_frame_format = "{frame_idx:06d}.png"
    working_dir = f"{base_folder}/working_dir/{test_id}/{video_id}_qp{qp_idx}"
    log_file = f"{log_dir}/encoding_{video_id}_qp{qp_idx}.log"
    cfg_file = f"{log_dir}/encoding_{video_id}_qp{qp_idx}.cfg"

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
    cfg["InnerCodec"] = 'NNVVC'
    cfg["num_workers"] = cpu_per_gpu_process
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

    vcmrs.log('\n'.join(map(str,cmd)))
    with open(cfg_file, 'w', newline='\n') as f:
      f.write('\n'.join(map(str,cmd)))

    tasks.append(cmd)
    log_files.append(log_file)

start_time = time.time()
rpool = utils.GPUPool(device_ids = cuda_devices, proc_per_dev = processes_per_gpu)
err = rpool.process_tasks(tasks)
vcmrs.log(f'error code: {err}')
vcmrs.log(f'Elapse: {time.time() - start_time}, GPUs {len(cuda_devices)}')
vcmrs.log('all done')

### Collect encoding times ###
times = utils.collect_coding_times(log_files, seq_order_list = list(config.seq_cfg[scenario].keys()))
vcmrs.log("Coding time by QP:")
vcmrs.log(times)
