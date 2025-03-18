#!/bin/bash
SPATH=$(dirname $(realpath ${BASH_SOURCE[0]}));
BDIR=$(realpath $SPATH);

cd $BDIR;
mkdir -p build;
cd build;
cmake ..;
make -j;
cd ..;

mkdir -p results;
cd results;

# no transpose
L="EE1-1.10_model0 slice scatternd_c_tf2 scatternd_hwc_with_conv_tf2 slice_hwc resize_bilinear_up2_tf2 resize_nearest_up2_tf2 reshape_m1_tf2";
for F in $L; do
  ../utest.sh ../models/${F}.onnx --no_transpose;
done


L="conv2d_4_8x8x4_k1x1s1,1_g1_p0,0 conv2d_4_8x8x4_k1x1s1,1_g4_p0,0 conv2d_4_8x8x4_k1x1s2,1_g1_p0,0 conv2d_4_8x8x4_k1x3s1,2_g1_p0,1 conv2d_4_8x8x4_k3x1s1,1_g1_p1,0 conv2d_4_8x8x4_k3x1s1,1_g4_p1,0 conv2d_4_8x8x4_k3x3s1,1_g1_p1,1 conv2d_4_8x8x4_k3x3s2,1_g1_p1,1 conv2d_4_8x8x4_k3x3s2,2_g1_p1,1 conv2d_4_8x8x4_k5x5s1,1_g1_p2,2 conv2d_4_8x8x4_k5x5s1,1_g4_p2,2 conv2d_4_8x8x4_k5x5s2,1_g1_p2,2 conv2d_4_8x9x4_k1x1s2,1_g1_p0,0 conv2d_4_8x9x4_k3x1s1,1_g4_p1,0 conv2d_4_8x9x4_k3x3s1,1_g4_p1,1 conv2d_4_8x9x4_k3x3s2,1_g1_p1,1 conv2d_4_8x9x4_k3x3s2,2_g1_p1,1 conv2d_4_9x8x4_k1x1s1,1_g1_p0,0 conv2d_4_9x8x4_k1x1s2,1_g1_p0,0 conv2d_4_9x8x4_k1x3s1,2_g1_p0,1 conv2d_4_9x8x4_k3x1s1,1_g1_p1,0 conv2d_4_9x8x4_k3x3s1,1_g1_p1,1 conv2d_4_9x8x4_k3x3s2,1_g1_p1,1 conv2d_4_9x8x4_k3x3s2,2_g1_p1,1 conv2d_4_9x8x4_k5x5s1,1_g1_p2,2 conv2d_4_9x8x4_k5x5s2,1_g1_p2,2 conv2d_4_9x9x4_k1x3s1,2_g1_p0,1 repeated_conv slice_pytorch slice_inf_pytorch slice_chw_pytorch prelu_multiple_alpha prelu_single_alpha scatternd_c_pytorch scatternd_hwc_with_conv_pytorch gridsample_bilinear gridsample_nearest gridsample_bilinear_conv gridsample_nearest_conv conv2dt_32_8x8x32_k3,3_s2,2_p1,1_op1,1 conv2dt_32_8x8x32_k4,4_s2,2_p1,1_op0,0 conv2dt_32_8x8x32_k5,5_s2,2_p2,2_op1,1 resize_bilinear_up2_pytorch resize_nearest_up2_pytorch resize_bilinear_up2_16x16x64_pytorch prelu_single_alpha_c32 prelu_multiple_alpha_c32 compare_less compare_greater where_constA_less where_constA_greater where_constB_less where_constB_greater maxpool_k2_s2 maxpool_k3_s3 global_maxpool conv3x1_1x3_s2_g4 clip reshape_1234 reshape_minusone mult_4dx4d mult_reshape_by_transpose_reshape attention_block_fixed reshape_withshape attention_block_dyn"
# mini_model_bb_mult 
for F in $L; do
  ../utest.sh ../models/${F}.onnx;
done

# test border peeling
function test_border() {
  rm -f $1.sadl;
  python3 ../../converter/main.py --input_onnx ../models/$1.onnx  --output $1.sadl > /dev/null
  if [ ! -f ${1}.sadl ]; then
   echo " [ERROR] $1 onnx to SADL conversion failed";
   exit -1;
  fi
  
  ../build/test_border_avx2 $1.sadl $2 $3 
  E=$?;
  if (( E == 0 )); then
   echo -e "   \e[32m[PASS]\e[0m $1 border"
  else
   echo -e "   \e[31m[FAIL]\e[0m $1 border"
   exit -1;
  fi;
}

test_border conv2d_8x8x1_g1_3s1x3s1x4 1 1
test_border conv2d_7x8x1_g1_3s1x3s1x4 1 1
test_border conv2d_8x7x1_g1_3s1x3s1x4 1 1
 


