#!/bin/bash
SPATH=$(dirname $(realpath ${BASH_SOURCE[0]}));
BDIR=$(realpath $SPATH);

if [ $# != 1 -a $# != 2 ]; then
 echo "[ERROR] utest.sh model.onnx [--no_transpose]"
 exit -1;
fi;
OPT="";
if [ $# == 2 ]; then
 OPT="$2";
fi;
 
if [ ! -f ${1} ]; then
 echo "[ERROR] no file $1"
 exit -1;
fi;

OPT2="--input_default_value=8"; # for graph with named inputs

NAME=$(basename $1);
BNAME=${NAME/.onnx/};

rm -f ${BNAME}.results;
python3 ${BDIR}/onnx_inference.py --input_onnx $1 --output ${BNAME}.results $OPT $OPT2 > /dev/null
if [ ! -f ${BNAME}.results ]; then
 echo " [ERROR] onnx inference failed for ${BNAME}";
 exit -1;
fi

rm -f ${BNAME}.sadl;
python3 ${BDIR}/../converter/main.py --input_onnx $1  --output ${BNAME}.sadl $OPT2 > /dev/null
if [ ! -f ${BNAME}.sadl ]; then
 echo " [ERROR] onnx to SADL conversion failed for ${BNAME}";
 exit -1;
fi

${BDIR}/build/test_scalar ${BNAME}.sadl ${BNAME}.results 0.001 > /dev/null
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m $BNAME scalar" 
else
  echo -e "   \e[31m[FAIL]\e[0m $BNAME scalar"
  exit -1;  
fi;

${BDIR}/build/test_avx2 ${BNAME}.sadl ${BNAME}.results 0.001 > /dev/null
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m $BNAME avx2" 
else
  echo -e "   \e[31m[FAIL]\e[0m $BNAME avx2"
  exit -1;  
fi;

${BDIR}/build/test_avx512 ${BNAME}.sadl ${BNAME}.results 0.001 > /dev/null
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m $BNAME avx512" 
else
  echo -e "   \e[31m[FAIL]\e[0m $BNAME avx512"
  exit -1;  
fi;

