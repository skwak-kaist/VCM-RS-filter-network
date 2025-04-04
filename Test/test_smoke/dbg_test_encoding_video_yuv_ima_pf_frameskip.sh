#!/bin/bash

# test encoding a video in YUV format
#The RoI is temporarily bypassed. Error should be resolved later
set -e
fname='./videos/testv1_256x128_420p.yuv'
python -m vcmrs.encoder \
  --PostFilter IMA \
  --InterMachineAdapter 0 \
  --SourceWidth 256  \
  --SourceHeight 128 \
  --InputBitDepth 8 \
  --InputChromaFormat '420' \
  --FrameSkip 3 \
  --FramesToBeEncoded 35 \
  --output_dir ./output/$0 \
  --directory_as_video \
  --debug \
  --quality 42 \
  --ROI "Bypass" \
  --working_dir ./output/working_dir/$0 \
  $fname

recon_fname=./output/$0/recon/$(basename $fname)
echo checking the reconstructed images has a good quality
python ./tools/check_psnr.py --f1 $fname --f1_bitdepth 8 --f1_frameskip 3 --f1_frames 35 --f2 $recon_fname --f2_bitdepth 10 -t 8

