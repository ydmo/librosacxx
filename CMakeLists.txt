cmake_minimum_required(VERSION 3.0.0)

project(rosacxx LANGUAGES CXX C)

option(BUILD_ROSACXX_TESTS                      "Build librosacxx test."                                                ON)
option(BUILD_ROSACXX_FFT_WITH_SINGLE_PRECISION  "Build librosacxx fft libraries using single precision(32bit-float)."   ON)

include_directories( ${CMAKE_CURRENT_LIST_DIR} )
include_directories( ${CMAKE_CURRENT_LIST_DIR}/../half/include/half )

set(
    ROSACXX_SRCS

    # Thrid-Parties ----------------------------------------------

    ## N-Dimension Array Library like numpy.
    ${CMAKE_CURRENT_LIST_DIR}/3rd/numcxx/numcxx.h
    ${CMAKE_CURRENT_LIST_DIR}/3rd/numcxx/alignmalloc.h
    ${CMAKE_CURRENT_LIST_DIR}/3rd/numcxx/ndarray.h
    ${CMAKE_CURRENT_LIST_DIR}/3rd/numcxx/pad.h

    ## FFT Libraries.
    ${CMAKE_CURRENT_LIST_DIR}/3rd/fft/fft.h
    ${CMAKE_CURRENT_LIST_DIR}/3rd/fft/fft.cc
    ${CMAKE_CURRENT_LIST_DIR}/3rd/fft/kiss/ext/vamp_kiss_fft_guts.h
    ${CMAKE_CURRENT_LIST_DIR}/3rd/fft/kiss/ext/vamp_kiss_fft_scalar.h
    ${CMAKE_CURRENT_LIST_DIR}/3rd/fft/kiss/ext/vamp_kiss_fft.h
    ${CMAKE_CURRENT_LIST_DIR}/3rd/fft/kiss/ext/vamp_kiss_fftr.h
    ${CMAKE_CURRENT_LIST_DIR}/3rd/fft/kiss/kiss_fft.h
    ${CMAKE_CURRENT_LIST_DIR}/3rd/fft/kiss/kiss_fft.cc

    # ROSA -------------------------------------------------------
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/rosacxx.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/rosacxx.cc
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/filters.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/util/utils.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/util/utils.cc
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
    add_compile_options(-std=c++11)
    add_compile_options(-pthread)
    find_package(GTest REQUIRED)
    add_executable(
        rosacxx_tests
        ${CMAKE_CURRENT_LIST_DIR}/tests/tests_all.cc
        ${CMAKE_CURRENT_LIST_DIR}/tests/tests_fft.cc
        ${CMAKE_CURRENT_LIST_DIR}/tests/tests_numcxx.cc
        ${CMAKE_CURRENT_LIST_DIR}/tests/tests_rosacxx.cc
        )
    target_link_libraries(rosacxx_tests PUBLIC rosacxx)
    set(GTEST_LIBRARIES gtest gtest_main pthread)
    target_link_libraries(rosacxx_tests PUBLIC ${GTEST_LIBRARIES})
endif()

if (BUILD_ROSACXX_FFT_WITH_SINGLE_PRECISION)
    add_definitions(-DROSACXX_FFT_PRECISION_F32=1)
    add_definitions(-DUSE_VAMP_KISS_FFT)
endif()

