#!/bin/bash
source_dir=`pwd`
external_dir=${source_dir}/external
mkdir -p external

# build METIS
cd ${external_dir}
wget https://karypis.github.io/glaros/files/sw/metis/metis-5.1.0.tar.gz
tar xfz metis-5.1.0.tar.gz
rm metis-5.1.0.tar.gz
cd metis-5.1.0
mkdir -p install
make config prefix=${external_dir}/metis-5.1.0/install
make -j 8
make install

# build GSL
cd ${external_dir}
wget https://ftp.gnu.org/gnu/gsl/gsl-2.7.1.tar.gz
tar xfz gsl-2.7.1.tar.gz
rm gsl-2.7.1.tar.gz
cd gsl-2.7.1
mkdir -p gsl
./configure --prefix=${source_dir}/external/gsl-2.7.1/gsl
make -j 8
make install

# build SZ3
cd ${external_dir}
git clone https://github.com/szcompressor/SZ3.git
cd SZ3
git reset --hard be68d645b2e1350adfbd61851c0886b38b876aa5
cp ${source_dir}/SZ3_CMakeLists.txt CMakeLists.txt
mkdir -p build 
mkdir -p install
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=${external_dir}/SZ3/install -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
make -j 8
make install

# build ZFP
cd ${external_dir}
git clone https://github.com/LLNL/zfp.git
cd zfp
mkdir -p build
mkdir -p install
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=${external_dir}/zfp/install -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
make -j 8
make install

# build FTK for evaluation of critical point preservation
cd ${external_dir}
git clone https://github.com/hguo/ftk.git
cd ftk
mkdir -p install
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=${external_dir}/ftk/install ..
make -j 8
make install

# build UMC
cd ${source_dir}
mkdir -p build
cd build
cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
make -j 8