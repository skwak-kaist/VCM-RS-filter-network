#!/bin/bash

# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

# install VCM-RS encoder+decoder package

# recommended environment
echo ' 
  Recommnended environment: 
    CUDA: 11.3
    python 3.8.13
    torch 1.12.0+cu113
    torchvision 0.13.0+cu113
'

bash dependencies_install.sh

set -e

#env_name=vcm
env_name=$(head -n 1 vcm_env_name.txt | xargs)
echo "Installing VCM-RS in conda environment:" $env_name

# Run setup_vcm_env.sh to setup VCM environment for conda. 
bash setup_vcm_env.sh $env_name

# activate vcm_decoder environment
eval "$(conda shell.bash hook)"
conda activate $env_name

echo "Checking Python version"
python --version

echo "Listing python packages before install:"
pip list

set -x

#pip install gdown

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
# downgrad pillow
pip install pillow==9.5.0

pip install Cython==3.0.0

pushd ./vcmrs/SpatialResample/models
git clone https://github.com/WongKinYiu/yolov7.git
rm -rf yolov7/requirements.txt
popd

# Temporal_Post_Hint
TEMPORAL_POST_HINT="True"
if [ "$TEMPORAL_POST_HINT" = "True" ]; then
  pip install einops
  pip install timm
  # ckpt_koniq10k.pt  
fi

# install python packages
pip install -r Requirements.txt

# install VCMRS
pip install --no-deps -e .

# compile VTM
pushd ./vcmrs/InnerCodec/VTM
mkdir -p build
cd build
cmake ..
make -j 8
popd

# Install the LIC intra codec package
pushd ./vcmrs/InnerCodec/NNManager/NNIntraCodec/LIC/e2evc
bash install.sh
popd

echo "Listing python packages after install:"
pip list

bash get_cpp_checksums.sh

# install testing image and video
pushd Test
python tools/gen_video.py # generate test video with resolution 256x128
python tools/gen_video.py --width 1920 --height 1080 --length 65
python tools/gen_video.py --width 640 --height 512 --length 5
python tools/gen_video.py --width 257 --height 128 --length 5
python tools/gen_image.py --width 637 --height 517 
python tools/gen_image.py --width 638 --height 517 
python tools/gen_image.py --width 1920 --height 1080 
#python tools/gen_image.py --width 2560 --height 1600
popd

echo "Finished intialling VCM-RS in conda environment:" $env_name
