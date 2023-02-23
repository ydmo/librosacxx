cmake_minimum_required(VERSION 3.0.0)

# --------------------------------------------------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------------------------------------------------

function(assign_source_group)
    foreach(_source IN ITEMS ${ARGN})
        if (IS_ABSOLUTE "${_source}")
            file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
        else()
            set(_source_rel "${_source}")
        endif()
        get_filename_component(_source_path "${_source_rel}" PATH)
        string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
        source_group("${_source_path_msvc}" FILES "${_source}")
    endforeach()
endfunction(assign_source_group)
 
function(my_add_library)
    foreach(_source IN ITEMS ${ARGN})
        assign_source_group(${_source})
    endforeach()
    add_library(${ARGV})
endfunction(my_add_library)

# --------------------------------------------------------------------------------------------------------------------------------
# Project
# --------------------------------------------------------------------------------------------------------------------------------

project(rosacxx LANGUAGES CXX C VERSION 1.1.0)

add_compile_options(-std=c++11)
# add_compile_options(-pthread)
add_compile_options(-fPIC)
add_compile_options(-shared)

option(BUILD_ROSACXX_TESTS                      "Build librosacxx test."                                                OFF)
option(BUILD_ROSACXX_TESTS_VISUALIZE            "Build librosacxx test with opencv visualize helper."                   OFF)
option(BUILD_ROSACXX_FFT_WITH_SINGLE_PRECISION  "Build librosacxx fft libraries using single precision(32bit-float)."   ON)
option(BUILD_STATIC_LIB                         "Build static libs."                                                    ON)

add_definitions(-DUSE_VAMP_KISS_FFT)
if (BUILD_ROSACXX_FFT_WITH_SINGLE_PRECISION)
    add_definitions(-DROSACXX_FFT_PRECISION_F32=1)
else()
    add_definitions(-DROSACXX_FFT_PRECISION_F32=0)
endif()

if(MSVC)
    message(STATUS "Compiler is msvc.")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    set(CMAKE_CXX_FLAGS_DEBUG          "-D__DEBUG=1  -Od /MDd -Zi")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-D__NDEBUG=1 -O2 /MD  -Zi")
    set(CMAKE_CXX_FLAGS_MINSIZEREL     "-D__NDEBUG=1 -Os /MD")
    set(CMAKE_CXX_FLAGS_RELEASE        "-D__NDEBUG=1 -O2 /MD")
    set(MY_VERSIONINFO_RC "${CMAKE_BINARY_DIR}/versioninfo.rc")
    configure_file("${CMAKE_CURRENT_LIST_DIR}/versioninfo.rc.in" "${MY_VERSIONINFO_RC}")
else()
    message(STATUS "Compiler is gcc/g++.")
    set(CMAKE_CXX_FLAGS_DEBUG          "-D__DEBUG=1  -O0 -g")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-D__NDEBUG=1 -O3 -g")
    set(CMAKE_CXX_FLAGS_MINSIZEREL     "-D__NDEBUG=1 -O3")
    set(CMAKE_CXX_FLAGS_RELEASE        "-D__NDEBUG=1 -O3")
    if (NOT CMAKE_BUILD_TYPE)
        set (CMAKE_BUILD_TYPE "Release")
        message(STATUS "No CMAKE_BUILD_TYPE set, defaulting to ${CMAKE_BUILD_TYPE}")
    endif ()
endif()

include_directories( ${CMAKE_CURRENT_LIST_DIR} )
include_directories( ${CMAKE_CURRENT_LIST_DIR}/3rd/googletest/ )
include_directories( ${CMAKE_CURRENT_LIST_DIR}/3rd/googletest/include/ )
include_directories( ${CMAKE_CURRENT_LIST_DIR}/3rd/googletest/src/ )

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
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/util/visualize.h
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

if (BUILD_STATIC_LIB)
    my_add_library(rosacxx STATIC ${ROSACXX_SRCS} ${MY_VERSIONINFO_RC})
    set_target_properties(rosacxx PROPERTIES OUTPUT_NAME "rosacxx")
    set_target_properties(rosacxx PROPERTIES CLEAN_DIRECT_OUTPUT 1)
else()
    my_add_library(rosacxx SHARED ${ROSACXX_SRCS} ${MY_VERSIONINFO_RC})
endif(BUILD_STATIC_LIB)

target_include_directories(rosacxx INTERFACE ${CMAKE_CURRENT_LIST_DIR}/rosacxx)

if (BUILD_ROSACXX_TESTS)

    add_definitions(-DROSACXX_TEST=1)

    set(CMAKE_CXX_FLAGS_DEBUG "-g -D__DEBUG=1")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -D__NDEBUG=1")
    set(GTEST_SRCS ${CMAKE_CURRENT_LIST_DIR}/3rd/googletest/src/gtest-all.cc)
    add_library(googletest STATIC ${GTEST_SRCS})
    target_include_directories(googletest INTERFACE ${CMAKE_CURRENT_LIST_DIR}/3rd/googletest/src)
    set_target_properties(googletest PROPERTIES OUTPUT_NAME "googletest")
    set_target_properties(googletest PROPERTIES CLEAN_DIRECT_OUTPUT 1)

    add_executable(
        rosacxx_tests
        ${CMAKE_CURRENT_LIST_DIR}/tests/tests_data.h
        ${CMAKE_CURRENT_LIST_DIR}/tests/tests_all.cc
        ${CMAKE_CURRENT_LIST_DIR}/tests/tests_fft.cc
        ${CMAKE_CURRENT_LIST_DIR}/tests/tests_numcxx.cc
        ${CMAKE_CURRENT_LIST_DIR}/tests/tests_rosacxx.cc
        )
    target_link_libraries(rosacxx_tests PUBLIC googletest rosacxx)

    if (BUILD_ROSACXX_TESTS_VISUALIZE)
        add_definitions(-DROSACXX_TESTS_VISUALIZE=1)
    endif()

endif()


