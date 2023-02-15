cmake_minimum_required(VERSION 3.15)
project(cl_test)

find_package(OpenCL REQUIRED)
add_executable(cl_test main.cpp)
target_link_libraries(cl_test PRIVATE OpenCL::OpenCL)
target_compile_definitions(cl_test PUBLIC -DCL_TARGET_OPENCL_VERSION=120)
