cmake_minimum_required(VERSION 3.0.0)
project(rosacxx LANGUAGES CXX C)
option(BUILD_ROSACXX_TESTS "Build librosacxx test." OFF)
include_directories( ${CMAKE_CURRENT_LIST_DIR}/rosacxx )
set(
    ROSACXX_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/rosacxx.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/rosacxx.cc
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/core/constantq.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/core/constantq.cc
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/feature/spectral.h
    ${CMAKE_CURRENT_LIST_DIR}/rosacxx/feature/spectral.cc
    )
add_library(rosacxx STATIC ${ROSACXX_SRCS})
target_include_directories(rosacxx INTERFACE ${CMAKE_CURRENT_LIST_DIR}/rosacxx)
set_target_properties(rosacxx PROPERTIES OUTPUT_NAME "rosacxx")
set_target_properties(rosacxx PROPERTIES CLEAN_DIRECT_OUTPUT 1)
if (BUILD_ROSACXX_TESTS)
    add_executable(
        rosacxx-tests
        ${CMAKE_CURRENT_LIST_DIR}/tests/tests_all.h
        ${CMAKE_CURRENT_LIST_DIR}/tests/tests_all.cc
        )
    target_link_libraries(rosacxx-tests PUBLIC rosacxx)
endif()
