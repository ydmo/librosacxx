cmake_minimum_required(VERSION 3.0.0)
# add_compile_options(-std=c++11)
project(gtest LANGUAGES CXX C)

include_directories( ${PROJECT_SOURCE_DIR}/src )

set(GTEST_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/gtest-all.cc
    )

add_library(gtest STATIC ${GTEST_SRCS})

target_include_directories(gtest INTERFACE ${CMAKE_CURRENT_LIST_DIR}/src)

set_target_properties(gtest PROPERTIES OUTPUT_NAME "")
set_target_properties(gtest PROPERTIES CLEAN_DIRECT_OUTPUT 1)
