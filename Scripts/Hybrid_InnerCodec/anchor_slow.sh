#!/bin/bash

# This script encode the whole VCM test datasets in slow mode for encoding time evaluation

eval "$(conda shell.bash hook)"

env_name=$(head -n 1 ../../vcm_env_name.txt | xargs)
echo $env_name
conda activate $env_name


set -e

# SFU dataset
#echo
#echo '#######################################################'
#echo
#python encode_sfu_AI.py
#
#echo
#echo '#######################################################'
#echo
#python encode_sfu_LD.py
#
#echo
#echo '#######################################################'
#echo
#python encode_sfu_RA.py

# TVD dataset
echo
echo '#######################################################'
echo
python encode_tvd_tracking_AI.py

echo
echo '#######################################################'
echo
python encode_tvd_tracking_LD.py

echo
echo '#######################################################'
echo
python encode_tvd_tracking_RA.py

# image datasets
echo
echo '#######################################################'
echo
python encode_openimages.py

echo
echo '#######################################################'
echo
python encode_flir.py

echo
echo '#######################################################'
echo
python encode_tvd_detseg.py


