# Unstructured data compression
Project: UMC

# Installation
git clone https://github.com/xuanwu02/unstructured_data_compression.git <br>
cd unstructured_data_compression <br>
sh build_script.sh <br>

# Run compression
./build/test/test_compress data/coordinates.dat data/connectivity.dat 1 2 2 1 2 data/momentumX.dat data/velocityX.dat 0 1e-2 2 <br>

Compressed files are data/momentumX.dat.umc and data/velocityX.dat.umc

# Run decompression
./build/test/test_decompress data/coordinates.dat data/connectivity.dat 1 2 2 1 2 data/momentumX.dat data/velocityX.dat <br>

Decompressed files are data/momentumX.dat.umc.out and data/velocityX.dat.umc.out
