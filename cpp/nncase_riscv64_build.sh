#!/bin/bash

set -x

export PATH=$PATH:/home/zhangyang/workspace/github/llm/pipeline/toolchain/Xuantie-900-gcc-linux-6.6.0-glibc-x86_64-V2.10.1/bin

build=nncase_riscv64_build
build_type=Release
# build_type=Debug
rm -rf ${build}
mkdir ${build}
pushd ${build}
cmake -DBUILD_NNCASE=1 \
      -DCMAKE_BUILD_TYPE=${build_type}           \
      -DCMAKE_INSTALL_PREFIX=install    \
      -DCMAKE_TOOLCHAIN_FILE=cmake/Riscv64.cmake \
      ..
make -j8
make install
cp install/melotts /home/share/nfsroot/k230/k230_llm/melotts_nncase
popd

# cp ${build}/demo /home/share/nfsroot/k230/k230_llm/pipeline


