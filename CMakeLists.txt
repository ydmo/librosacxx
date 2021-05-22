cmake_minimum_required(VERSION 3.0.0)

add_compile_options(-std=c++11)
add_compile_options(-pthread)

project(rosacxx LANGUAGES CXX C)

option(BUILD_ROSACXX_TESTS                      "Build librosacxx test."                                                ON)
option(BUILD_ROSACXX_FFT_WITH_SINGLE_PRECISION  "Build librosacxx fft libraries using single precision(32bit-float)."   ON)

include_directories( ${CMAKE_CURRENT_LIST_DIR} )
include_directories( ${CMAKE_CURRENT_LIST_DIR}/../googletest/ )
include_directories( ${CMAKE_CURRENT_LIST_DIR}/../googletest/include/ )
include_directories( ${CMAKE_CURRENT_LIST_DIR}/../googletest/src/ )

set(
    ROSACXX_SRCS

    # Thrid-Parties ----------------------------------------------

    ## Half Float
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/half/half.h

    ## Complex Data
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/complex/complex.h

    ## N-Dimension Array Library like numpy.
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/numcxx/numcxx.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/numcxx/alignmalloc.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/numcxx/ndarray.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/numcxx/pad.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/numcxx/utils.h

    ## Efficient time-series resampling library like resampy.
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/resamcxx/core.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/resamcxx/data.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/resamcxx/filters.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/resamcxx/interpn.h

    ## FFT Libraries.
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/fft/fft.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/fft/fft.cc
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/fft/kiss/ext/vamp_kiss_fft_guts.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/fft/kiss/ext/vamp_kiss_fft_scalar.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/fft/kiss/ext/vamp_kiss_fft.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/fft/kiss/ext/vamp_kiss_fftr.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/fft/kiss/kiss_fft.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/fft/kiss/kiss_fft.cc

    # ROSA -------------------------------------------------------
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/rosacxx.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/rosacxx.cc
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/filters.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/util/utils.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/util/utils.cc
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/core/audio.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/core/audio.cc
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/core/constantq.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/core/constantq.cc
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/core/convert.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/core/convert.cc
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/core/pitch.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/core/pitch.cc
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/core/fft.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/core/fft.cc
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/core/spectrum.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/core/spectrum.cc

    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/feature/spectral.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/feature/spectral.cc

    )
add_library(rosacxx STATIC ${ROSACXX_SRCS})
target_include_directories(rosacxx INTERFACE ${CMAKE_CURRENT_LIST_DIR}/rosacxx)
set_target_properties(rosacxx PROPERTIES OUTPUT_NAME "rosacxx")
set_target_properties(rosacxx PROPERTIES CLEAN_DIRECT_OUTPUT 1)

if (BUILD_ROSACXX_TESTS)
    set(GTEST_SRCS
        ${CMAKE_CURRENT_LIST_DIR}/../googletest/src/gtest-all.cc
        )
    add_library(rosacxx_gtest STATIC ${GTEST_SRCS})
    set_target_properties(rosacxx_gtest PROPERTIES OUTPUT_NAME "rosacxx_gtest")
    set_target_properties(rosacxx_gtest PROPERTIES CLEAN_DIRECT_OUTPUT 1)
    add_executable(
        rosacxx_tests
        ${CMAKE_CURRENT_LIST_DIR}/tests/tests_data.h
        ${CMAKE_CURRENT_LIST_DIR}/tests/tests_all.cc
        ${CMAKE_CURRENT_LIST_DIR}/tests/tests_fft.cc
        ${CMAKE_CURRENT_LIST_DIR}/tests/tests_numcxx.cc
        ${CMAKE_CURRENT_LIST_DIR}/tests/tests_rosacxx.cc
        )
    target_link_libraries(rosacxx_tests PUBLIC rosacxx rosacxx_gtest)
endif()

add_definitions(-DUSE_VAMP_KISS_FFT)
if (BUILD_ROSACXX_FFT_WITH_SINGLE_PRECISION)
    add_definitions(-DROSACXX_FFT_PRECISION_F32=1)
else()
    add_definitions(-DROSACXX_FFT_PRECISION_F32=0)
endif()
