cmake_minimum_required(VERSION 3.15)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
project(ort_deploy LANGUAGES C CXX)
include(cmake/common.cmake)

print_cache_vars()

set(ORT_ROOT /home/wangyushun/novauto/official_ort)
set(ORT_INC_DIR ${ORT_ROOT} ${ORT_ROOT}/include)
find_library(ORT_LIB NAMES onnxruntime PATHS ${ORT_ROOT}/build/install/usr/local/lib/)

# build all targets
add_subdirectory(src)
add_subdirectory(test)

