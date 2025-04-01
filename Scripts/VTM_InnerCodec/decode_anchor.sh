#!/bin/bash

decode_folder=decode_without_colorize

./decode_sfu.sh SFU_AI_e2e ${decode_folder}
./decode_sfu.sh SFU_AI_inner ${decode_folder}
./decode_sfu.sh SFU_LD_e2e ${decode_folder}
./decode_sfu.sh SFU_LD_inner ${decode_folder}
./decode_sfu.sh SFU_RA_e2e ${decode_folder}
./decode_sfu.sh SFU_RA_inner ${decode_folder}


