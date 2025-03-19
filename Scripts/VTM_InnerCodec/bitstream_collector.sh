#!/bin/bash

datasets="TVD_tracking SFU"
scenarios="inner e2e"
encoding_configs="RA LD AI"

for dataset in $datasets; do
    for scenario in $scenarios; do
        for encoding_config in $encoding_configs; do
            # dataset_encoding_config_scenario 폴더에 들어가서 bitstream 폴더를 anchor bitstream 폴더로 복사

            mkdir anchor_bitstream/${dataset}_${encoding_config}_${scenario}

            cp -r ${dataset}_${encoding_config}_${scenario}/bitstream anchor_bitstream/${dataset}_${encoding_config}_${scenario}/

        done
    done
done