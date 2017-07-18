cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)
# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-fPIC -std=c++11 ${CMAKE_CXX_FLAGS} -march=native")
set(CUDA_NVCC_FLAGS "-I ${Tensorflow_INCLUDE_DIRS} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC")

# build the actual operation which can be used directory
include_directories(${Tensorflow_INCLUDE_DIRS})
cuda_add_library(correlation SHARED correlation.cc correlation_gpu.cc.cu)
target_link_libraries(correlation cudart)

# build the gradient operation which is used in correlation_grad.py
# to register it
include_directories(${Tensorflow_INCLUDE_DIRS})
#add_library(correlation_grad SHARED correlation_grad.cc)
cuda_add_library(correlation_grad SHARED correlation_grad.cc correlation_grad_gpu.cc.cu)
target_link_libraries(correlation_grad cudart correlation)
