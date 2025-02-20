

build=x86_64_build
build_type=Release
# build_type=Debug
rm -rf ${build}
mkdir -p ${build}

pushd ${build}
cmake ..  \
  -DBUILD_ONNX=1 \
  -DCMAKE_INSTALL_PREFIX=install \
  -DCMAKE_BUILD_TYPE=${build_type}
make -j8
make install

pushd install
models=../../../models
./melotts -e ${models}/encoder-zh.onnx \
          -d ${models}/decoder-zh.onnx \
          -l ${models}/lexicon.txt \
          -t ${models}/tokens.txt \
          --g ${models}/g-zh_mix_en.bin
popd
popd


