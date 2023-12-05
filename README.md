# Unstructured Data Compression

# Installation
sh build_script.sh <br>
Dependancies: METIS, GSL, ZSTD, SZ3

# Run compression
./build/test/test_compress data/coordinates.dat data/connectivity.dat 1 2 2 1 2 data/momentumX.dat data/velocityX.dat 0 1e-2 2
Compressed files are data/momentumX.dat.umc and data/velocityX.dat.umc

# Run decompression
./build/test/test_decompress data/coordinates.dat data/connectivity.dat 1 2 2 1 2 data/momentumX.dat data/velocityX.dat
Decompressed files are data/momentumX.dat.umc.out and data/velocityX.dat.umc.out
