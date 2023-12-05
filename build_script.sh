#!/bin/bash
src_dir=`pwd`

# build UMC
cd ${src_dir}
mkdir build
cd build
cmake ..
make -j 4