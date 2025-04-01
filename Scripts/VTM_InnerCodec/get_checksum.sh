#!/bin/bash

# this script calculates md5 checksum of output bitstreams and reconstructions

#cd output/anchor_bitstream

#find -L SFU_* TVD_* -name '*.bin' -exec md5sum {} \; | sort -k 2 > anchor_bitstream.chk

cd output/decode_without_colorize

find -L SFU_* TVD_*  -name '*.yuv' -exec md5sum {} \; | sort -k 2 > anchor_recon.chk

echo $0 completeds

