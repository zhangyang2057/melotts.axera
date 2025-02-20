
#!/bin/bash
nncaseruntime=`pwd`/3rd_party/nncase/x86_64/



build=nncase_x86_64_build
build_type=Release
# build_type=Debug
rm -rf ${build}
mkdir -p ${build}

pushd ${build}
cmake ..  \
  -DBUILD_NNCASE=1 \
  -DCMAKE_INSTALL_PREFIX=install \
  -DCMAKE_BUILD_TYPE=${build_type}
make -j8
make install

pushd install
export PATH=$PATH:${nncaseruntime}/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${nncaseruntime}/lib
models=../../../models
bin=./melotts
# bin="gdb --args ./melotts"
${bin} -e ${models}/encoder-zh.kmodel \
       -d ${models}/decoder-zh.kmodel \
       -l ${models}/lexicon.txt \
       -t ${models}/tokens.txt \
       --g ${models}/g-zh_mix_en.bin
popd
popd