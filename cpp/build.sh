
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
./melotts
popd