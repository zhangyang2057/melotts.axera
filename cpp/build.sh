
rm -rf install build
mkdir -p build

pushd build
cmake ..  \
  -DCMAKE_INSTALL_PREFIX=../install \
  -DCMAKE_BUILD_TYPE=Release
make -j4
make install
popd

pushd install
./melotts -e ../../models/encoder-zh.onnx \
          -d ../../models/decoder-zh.onnx \
          -l ../../models/lexicon.txt \
          -t ../../models/tokens.txt \
          --g ../../models/g-zh_mix_en.bin
popd
