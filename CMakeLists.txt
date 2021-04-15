cmake_minimum_required(VERSION 3.0.0)
project(rosacxx LANGUAGES CXX C)
include_directories( ${CMAKE_CURRENT_LIST_DIR}/rosacxx )
set(
    ROSACXX_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/rosacxx.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/rosacxx.cc
    )
add_library(rosacxx STATIC ${ROSACXX_SRCS})
target_include_directories(rosacxx INTERFACE ${CMAKE_CURRENT_LIST_DIR}/rosacxx)
set_target_properties(rosacxx PROPERTIES OUTPUT_NAME "rosacxx")
set_target_properties(rosacxx PROPERTIES CLEAN_DIRECT_OUTPUT 1)
