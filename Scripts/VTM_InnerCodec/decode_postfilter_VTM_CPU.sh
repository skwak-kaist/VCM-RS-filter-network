#!/bin/bash

# Decode in CPU (postfilter)

filter_mode=rgb_post_vtm_multi_qp_finetuned
decode_folder=decode_postfilter_vtm_bdr_off_cpu

./decode_sfu_postfilter.sh SFU_AI_e2e ${filter_mode} ${decode_folder}
./decode_sfu_postfilter.sh SFU_AI_inner ${filter_mode} ${decode_folder}
./decode_sfu_postfilter.sh SFU_LD_e2e ${filter_mode} ${decode_folder}
./decode_sfu_postfilter.sh SFU_LD_inner ${filter_mode} ${decode_folder}
./decode_sfu_postfilter.sh SFU_RA_e2e ${filter_mode} ${decode_folder}
./decode_sfu_postfilter.sh SFU_RA_inner ${filter_mode} ${decode_folder}

./decode_tvd_tracking_postfilter.sh TVD_tracking_AI_e2e ${filter_mode} ${decode_folder}
./decode_tvd_tracking_postfilter.sh TVD_tracking_AI_inner ${filter_mode} ${decode_folder}
./decode_tvd_tracking_postfilter.sh TVD_tracking_LD_e2e ${filter_mode} ${decode_folder}
./decode_tvd_tracking_postfilter.sh TVD_tracking_LD_inner ${filter_mode} ${decode_folder}
./decode_tvd_tracking_postfilter.sh TVD_tracking_RA_e2e ${filter_mode} ${decode_folder}
./decode_tvd_tracking_postfilter.sh TVD_tracking_RA_inner ${filter_mode} ${decode_folder}

