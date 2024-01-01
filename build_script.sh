#!/bin/bash
source_dir=`pwd`
external_dir=${source_dir}/external
mkdir -p external

# download METIS
cd ${external_dir}
git clone https://github.com/KarypisLab/METIS.git

# build GSL
curl -O https://ftp.gnu.org/gnu/gsl/gsl-2.7.1.tar.gz
tar xfz gsl-2.7.1.tar.gz
cd gsl-2.7.1
mkdir -p gsl
./configure --prefix=${source_dir}/external/gsl-2.7.1/gsl
make -j 8
make install

# build SZ3
cd ${external_dir}
git clone https://github.com/szcompressor/SZ3.git
cd SZ3
cp ${source_dir}/SZ3_CMakeLists.txt CMakeLists.txt
mkdir -p build 
mkdir -p install
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=${external_dir}/SZ3/install -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
make -j 8
make install

# build UMC
cd ${source_dir}
mkdir -p build
cd build
cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
make -j 8