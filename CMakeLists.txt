cmake_minimum_required(VERSION 3.7)
project(IP_particle_fluid_sim)
set(CMAKE_CXX_STANDARD 11)

# Fix Windows path environment variable.
STRING(REGEX REPLACE "\\\\" "/" OCL_ROOT $ENV{OCL_ROOT})

# Set required OpenCL CMake build options.
set(OpenCL_INCLUDE_DIR ${OCL_ROOT}/include/)
set(OpenCL_LIBRARY ${OCL_ROOT}/lib/x86/OpenCL.lib)

# Add OpenCL stuff.
include_directories($ENV{OCL_ROOT}/include/)
link_libraries($ENV{OCL_ROOT}/lib/x86/OpenCL.lib)
find_package(OpenCL REQUIRED)

include_directories($ENV{OCL_ROOT}/include/CL/)

# Add clEnqueueTask target.
set(clEnqueueTask_SOURCE_FILES
        clEnqueueTask/host.c
        clEnqueueTask/propagate.cl)
add_executable(clEnqueueTask ${clEnqueueTask_SOURCE_FILES})

# Add clEnqueueNDRangeKernel target.
set(clEnqueueNDRangeKernel_SOURCE_FILES
        clEnqueueNDRangeKernel/host.c
        clEnqueueNDRangeKernel/propagate.cl)
add_executable(clEnqueueNDRangeKernel ${clEnqueueNDRangeKernel_SOURCE_FILES})