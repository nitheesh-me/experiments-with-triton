cmake_minimum_required(VERSION 3.20)
project(triton-analysis)

# Find the installed MLIR.
# This assumes that MLIR was installed in a path discoverable by CMake,
# or that you provide its location with -DMLIR_DIR.
find_package(MLIR REQUIRED CONFIG)

# This includes the AddMLIR.cmake module which provides helper functions.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_subdirectory(tools)