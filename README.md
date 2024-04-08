# Unstructured data compression
Project: UMC

# Installation
git clone https://github.com/xuanwu02/umc.git <br>
cd umc <br>
sh build_script.sh <br>

# Run compression and decompression

## Compression
./build/test/test_compress data/coordinates.dat data/connectivity.dat 1 2 2 1 2 data/momentumX.dat data/velocityX.dat 0 1e-2 2

Compressed files are data/momentumX.dat.umc and data/velocityX.dat.umc

## Decompression
./build/test/test_decompress data/coordinates.dat data/connectivity.dat 1 2 2 1 2 data/momentumX.dat data/velocityX.dat

Decompressed files are data/momentumX.dat.umc.out and data/velocityX.dat.umc.out

# Enabling critical point preservation
The LES data file "velocity.dat" contains turbulent velovity in three dimensions. Separate data files are "vx.dat", "vy.dat" and "vz.dat". LES dataset is not included in this repo due to size constraints.

## Compression
./build/test/test_comp_decomp_cp_preserve coordinates.dat connectivity.dat 0 3 1 1 velocity.dat 0.05

Compressed file is velocity.dat.umc

## Decompression
./build/test/test_comp_decomp_cp_preserve coordinates.dat connectivity.dat 0 3 2 1 3 vx.dat vy.dat vz.dat velocity.dat.umc

Decompressed files are vx.dat.out, vy.dat.out and vz.dat.out

## Evaluation of critical point preservation using FTK
./build/test/cp_extraction_3d_unstructured coordinates.dat connectivity.dat vx.dat vy.dat vz.dat
