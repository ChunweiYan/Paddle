set -ex

mkdir -p build
cmake .. -DWITH_FLUID_ONLY=ON \
      -DWITH_GPU=ON \
      -DWITH_MKLDNN=OFF \
      -DWITH_TESTING=ON

make -j20
