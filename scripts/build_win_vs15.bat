@REM make sdk build dir

mkdir .\\libs-vs15

@REM build

call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\VC\Auxiliary\Build\vcvars64.bat"
mkdir .\\libs-vs15\\proj_x64
cd .\\libs-vs15\\proj_x64
cmake ..\\.. -G "Visual Studio 15 2017" -A x64 -DBUILD_ROSACXX_TESTS=OFF -DBUILD_ROSACXX_FFT_WITH_SINGLE_PRECISION=ON -DBUILD_STATIC_LIB=ON
devenv rosacxx.sln /Rebuild Debug   /Out log.txt
devenv rosacxx.sln /Rebuild Release /Out log.txt
cd ..\\..

call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars32.bat"
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\VC\Auxiliary\Build\vcvars32.bat"
mkdir .\\libs-vs15\\proj_x86
cd .\\libs-vs15\\proj_x86
cmake ..\\.. -G "Visual Studio 15 2017" -A Win32 -DBUILD_ROSACXX_TESTS=OFF -DBUILD_ROSACXX_FFT_WITH_SINGLE_PRECISION=ON -DBUILD_STATIC_LIB=ON
devenv rosacxx.sln /Rebuild Debug   /Out log.txt
devenv rosacxx.sln /Rebuild Release /Out log.txt
cd ..\\..

@REM copy include headers

mkdir .\\libs-vs15\\include
mkdir .\\libs-vs15\\include\\rosacxx
mkdir .\\libs-vs15\\include\\rosacxx\\complex
mkdir .\\libs-vs15\\include\\rosacxx\\core
mkdir .\\libs-vs15\\include\\rosacxx\\feature
mkdir .\\libs-vs15\\include\\rosacxx\\half
mkdir .\\libs-vs15\\include\\rosacxx\\numcxx
mkdir .\\libs-vs15\\include\\rosacxx\\resamcxx
mkdir .\\libs-vs15\\include\\rosacxx\\util
copy .\\rosacxx\\rosacxx.h                    .\\libs-vs15\\include\\rosacxx\\rosacxx.h
copy .\\rosacxx\\filters.h                    .\\libs-vs15\\include\\rosacxx\\filters.h
copy .\\rosacxx\\util\\utils.h                .\\libs-vs15\\include\\rosacxx\\util\\utils.h
copy .\\rosacxx\\resamcxx\\interpn.h          .\\libs-vs15\\include\\rosacxx\\resamcxx\\interpn.h
copy .\\rosacxx\\resamcxx\\filters.h          .\\libs-vs15\\include\\rosacxx\\resamcxx\\filters.h
copy .\\rosacxx\\resamcxx\\data.h             .\\libs-vs15\\include\\rosacxx\\resamcxx\\data.h
copy .\\rosacxx\\resamcxx\\data_kaiser_fast.h .\\libs-vs15\\include\\rosacxx\\resamcxx\\data_kaiser_fast.h
copy .\\rosacxx\\resamcxx\\data_kaiser_best.h .\\libs-vs15\\include\\rosacxx\\resamcxx\\data_kaiser_best.h
copy .\\rosacxx\\resamcxx\\core.h             .\\libs-vs15\\include\\rosacxx\\resamcxx\\core.h
copy .\\rosacxx\\numcxx\\utils.h              .\\libs-vs15\\include\\rosacxx\\numcxx\\utils.h
copy .\\rosacxx\\numcxx\\pad.h                .\\libs-vs15\\include\\rosacxx\\numcxx\\pad.h
copy .\\rosacxx\\numcxx\\numcxx.h             .\\libs-vs15\\include\\rosacxx\\numcxx\\numcxx.h
copy .\\rosacxx\\numcxx\\ndarray.h            .\\libs-vs15\\include\\rosacxx\\numcxx\\ndarray.h
copy .\\rosacxx\\numcxx\\alignmalloc.h        .\\libs-vs15\\include\\rosacxx\\numcxx\\alignmalloc.h
copy .\\rosacxx\\half\\half.h                 .\\libs-vs15\\include\\rosacxx\\half\\half.h
copy .\\rosacxx\\feature\\spectral.h          .\\libs-vs15\\include\\rosacxx\\feature\\spectral.h
copy .\\rosacxx\\core\\spectrum.h             .\\libs-vs15\\include\\rosacxx\\core\\spectrum.h
copy .\\rosacxx\\core\\resample.h             .\\libs-vs15\\include\\rosacxx\\core\\resample.h
copy .\\rosacxx\\core\\pitch.h                .\\libs-vs15\\include\\rosacxx\\core\\pitch.h
copy .\\rosacxx\\core\\fft.h                  .\\libs-vs15\\include\\rosacxx\\core\\fft.h
copy .\\rosacxx\\core\\convert.h              .\\libs-vs15\\include\\rosacxx\\core\\convert.h
copy .\\rosacxx\\core\\constantq.h            .\\libs-vs15\\include\\rosacxx\\core\\constantq.h
copy .\\rosacxx\\core\\audio.h                .\\libs-vs15\\include\\rosacxx\\core\\audio.h
copy .\\rosacxx\\complex\\complex.h           .\\libs-vs15\\include\\rosacxx\\complex\\complex.h

@REM copy libs  

mkdir .\\libs-vs15\\lib
mkdir .\\libs-vs15\\lib\\x64
mkdir .\\libs-vs15\\lib\\x64\\Release
mkdir .\\libs-vs15\\lib\\x64\\Debug
mkdir .\\libs-vs15\\lib\\x86
mkdir .\\libs-vs15\\lib\\x86\\Release
mkdir .\\libs-vs15\\lib\\x86\\Debug

copy .\\libs-vs15\\proj_x64\\Release\\rosacxx.dll .\\libs-vs15\\lib\\x64\\Release
copy .\\libs-vs15\\proj_x64\\Release\\rosacxx.lib .\\libs-vs15\\lib\\x64\\Release
copy .\\libs-vs15\\proj_x64\\Debug\\rosacxx.dll   .\\libs-vs15\\lib\\x64\\Debug
copy .\\libs-vs15\\proj_x64\\Debug\\rosacxx.lib   .\\libs-vs15\\lib\\x64\\Debug
copy .\\libs-vs15\\proj_x64\\Debug\\rosacxx.pdb   .\\libs-vs15\\lib\\x64\\Debug

copy .\\libs-vs15\\proj_x86\\Release\\rosacxx.dll .\\libs-vs15\\lib\\x86\\Release
copy .\\libs-vs15\\proj_x86\\Release\\rosacxx.lib .\\libs-vs15\\lib\\x86\\Release
copy .\\libs-vs15\\proj_x86\\Debug\\rosacxx.dll   .\\libs-vs15\\lib\\x86\\Debug
copy .\\libs-vs15\\proj_x86\\Debug\\rosacxx.lib   .\\libs-vs15\\lib\\x86\\Debug
copy .\\libs-vs15\\proj_x86\\Debug\\rosacxx.pdb   .\\libs-vs15\\lib\\x86\\Debug

pause