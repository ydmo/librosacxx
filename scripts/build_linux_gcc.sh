echo "[build] remove build targets"
rm -rf ./libs-gcc/*

echo "[build] create directories"
mkdir ./libs-gcc
mkdir ./libs-gcc/include
mkdir ./libs-gcc/include/rosacxx
mkdir ./libs-gcc/include/rosacxx/complex
mkdir ./libs-gcc/include/rosacxx/core
mkdir ./libs-gcc/include/rosacxx/feature
mkdir ./libs-gcc/include/rosacxx/half
mkdir ./libs-gcc/include/rosacxx/numcxx
mkdir ./libs-gcc/include/rosacxx/resamcxx

echo "[build] cp headers"
cp ./rosacxx/rosacxx.h                    ./libs-gcc/include/rosacxx/rosacxx.h
cp ./rosacxx/filters.h                    ./libs-gcc/include/rosacxx/filters.h
cp ./rosacxx/util/utils.h                ./libs-gcc/include/rosacxx/util/utils.h
cp ./rosacxx/resamcxx/interpn.h          ./libs-gcc/include/rosacxx/resamcxx/interpn.h
cp ./rosacxx/resamcxx/filters.h          ./libs-gcc/include/rosacxx/resamcxx/filters.h
cp ./rosacxx/resamcxx/data.h             ./libs-gcc/include/rosacxx/resamcxx/data.h
cp ./rosacxx/resamcxx/data_kaiser_fast.h ./libs-gcc/include/rosacxx/resamcxx/data_kaiser_fast.h
cp ./rosacxx/resamcxx/data_kaiser_best.h ./libs-gcc/include/rosacxx/resamcxx/data_kaiser_best.h
cp ./rosacxx/resamcxx/core.h             ./libs-gcc/include/rosacxx/resamcxx/core.h
cp ./rosacxx/numcxx/utils.h              ./libs-gcc/include/rosacxx/numcxx/utils.h
cp ./rosacxx/numcxx/pad.h                ./libs-gcc/include/rosacxx/numcxx/pad.h
cp ./rosacxx/numcxx/numcxx.h             ./libs-gcc/include/rosacxx/numcxx/numcxx.h
cp ./rosacxx/numcxx/ndarray.h            ./libs-gcc/include/rosacxx/numcxx/ndarray.h
cp ./rosacxx/numcxx/alignmalloc.h        ./libs-gcc/include/rosacxx/numcxx/alignmalloc.h
cp ./rosacxx/half/half.h                 ./libs-gcc/include/rosacxx/half/half.h
cp ./rosacxx/feature/spectral.h          ./libs-gcc/include/rosacxx/feature/spectral.h
cp ./rosacxx/core/spectrum.h             ./libs-gcc/include/rosacxx/core/spectrum.h
cp ./rosacxx/core/resample.h             ./libs-gcc/include/rosacxx/core/resample.h
cp ./rosacxx/core/pitch.h                ./libs-gcc/include/rosacxx/core/pitch.h
cp ./rosacxx/core/fft.h                  ./libs-gcc/include/rosacxx/core/fft.h
cp ./rosacxx/core/convert.h              ./libs-gcc/include/rosacxx/core/convert.h
cp ./rosacxx/core/constantq.h            ./libs-gcc/include/rosacxx/core/constantq.h
cp ./rosacxx/core/audio.h                ./libs-gcc/include/rosacxx/core/audio.h
cp ./rosacxx/complex/complex.h           ./libs-gcc/include/rosacxx/complex/complex.h

echo "[build] cd to build"
cd libs-gcc

echo "[build] cmake"
cmake .. -DBUILD_ROSACXX_TESTS=OFF -DBUILD_ROSACXX_FFT_WITH_SINGLE_PRECISION=ON -DBUILD_STATIC_LIB=OFF

echo "[build] make -j4"
make -j4