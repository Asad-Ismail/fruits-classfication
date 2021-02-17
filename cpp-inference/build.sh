mkdir build && cd build
cmake .. -DTFLITE_ENABLE_GPU=ON
cmake --build . -j 4
cd ..
