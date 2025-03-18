#!/bin/bash

# this script calculates md5 checksums of C++ source files

pushd vcmrs

find -L InnerCodec \( -name "*.cpp" -o -name "*.cxx" -o -name "*.hpp" -o -name "*.cc" -o -name "*.hh" -o -name "*.c" -o -name "*.h" \) -exec md5sum -b {} \; | sort -k 2 > cpp_sources.chk

popd

echo $0 completed

