#!/bin/bash


echo "[INFO] BUILD SADL SAMPLE"
# build sample
mkdir -p sample_test;
cd sample_test;
cmake -DCMAKE_BUILD_TYPE=Release ../sample
make
echo ""

echo "[INFO] TF2 -> ONNX -> SADL"
# TF2
python3 ../sample/tf2.py 2>/dev/null # output a tf2.onnx
python3 ../converter/main.py --input_onnx tf2.onnx --output tf2.sadl 
./sample_simd512 tf2.sadl 
echo ""

echo "[INFO] PYTORCH -> ONNX -> SADL"
# torch
python3 ../sample/pytorch.py 2>/dev/null # output a pytorch.onnx
python3 ../converter/main.py --input_onnx pytorch.onnx --output pytorch.sadl 
./sample_simd512 pytorch.sadl 
echo ""

echo "[INFO] DEBUG MODEL"
./debug_model pytorch.sadl > debug_model.log
echo "see debug_model.log"
echo ""

echo "[INFO] COUNT MAC"
./count_mac pytorch.sadl

echo "[INFO] WRITE INT16 MODEL"
echo  "0 15    1 8 2 0 3 8   6 8 7 0 8 8    10 9 11 0 12 8   14 8 15 1 16 8    20 8 21 0 22 9   24 8 25 0 26 8     28 8 29 0 30 8  34 8 35 0 36 8  38 0 39 40 0 0 42 8  43 0 44 8" | ./naive_quantization pytorch.sadl pytorch_int16.sadl;

echo "[INFO] QUANTIZATION CONSISTENCY"
# Usage: quantization_test model_float.sadl model_int16.sadl inputs_range shift max_error
python3 ../converter/main.py --input_onnx ../utests/models/quantization_test.onnx --output quantization_test.sadl
echo "0 10    1 10 2 0 3 10" | ./naive_quantization quantization_test.sadl quantization_test_int16.sadl;
./quantization_test quantization_test.sadl quantization_test_int16.sadl 1024.0 0 3
echo "0 10    1 9 2 0 3 9" | ./naive_quantization quantization_test.sadl quantization_test_int16.sadl;
./quantization_test quantization_test.sadl quantization_test_int16.sadl 1024.0 1 6
echo ""

echo "[INFO] BUILD SADL SAMPLE SPARSE"
# build sample
mkdir -p sparse;
cd sparse;
cmake -DCMAKE_BUILD_TYPE=Release -DSPARSE_MATMULT_SUPPORT=1 ../../sample
make sample_simd512 debug_model
echo ""

echo "[INFO] PYTORCH -> ONNX -> SADL"
# torch
python3 ../../sample/pytorch_matmult.py 2>/dev/null # output a pytorch.onnx
python3 ../../converter/main.py --input_onnx pytorch_matmult.onnx --output pytorch_matmult.sadl 
echo ""

echo "[INFO] CHECK sparsity"
./debug_model pytorch_matmult.sadl 
echo ""

echo "[INFO] INFERENCE not sparse"
./sample_simd512 pytorch_matmult.sadl 
echo ""

echo "[INFO] INFERENCE sparse"
./sample_simd512 pytorch_matmult.sadl 
echo ""
cd ..;


if [ -f tf2.sadl -a -f pytorch.sadl \
     -a -f sample_generic -a -f sample_simd256 -a -f sample_simd512 \
     -a -f count_mac \
     -a -f debug_model \
     -a -f naive_quantization \
     -a -f quantization_test \
     -a -f sparse/pytorch_matmult.sadl ]; then
 echo "[INFO] all build OK"
 exit 0;
else
 exit 1;
fi;
