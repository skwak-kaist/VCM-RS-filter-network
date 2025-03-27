#!/bin/bash

# Decode in CPU (anchor)

./decode_sfu.sh SFU_AI_e2e decode_anchor_cpu
./decode_sfu.sh SFU_AI_inner decode_anchor_cpu
./decode_sfu.sh SFU_LD_e2e decode_anchor_cpu
./decode_sfu.sh SFU_LD_inner decode_anchor_cpu
./decode_sfu.sh SFU_RA_e2e decode_anchor_cpu
./decode_sfu.sh SFU_RA_inner decode_anchor_cpu

./decode_tvd_tracking.sh TVD_tracking_AI_e2e decode_anchor_cpu
./decode_tvd_tracking.sh TVD_tracking_AI_inner decode_anchor_cpu
./decode_tvd_tracking.sh TVD_tracking_LD_e2e decode_anchor_cpu
./decode_tvd_tracking.sh TVD_tracking_LD_inner decode_anchor_cpu
./decode_tvd_tracking.sh TVD_tracking_RA_e2e decode_anchor_cpu
./decode_tvd_tracking.sh TVD_tracking_RA_inner decode_anchor_cpu


