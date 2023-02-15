#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <vector> 


const char* kernel_src =
"__kernel void reduction_vector(__global float4* data, __local float4* partial_sums, __global float* output)"
"{                                                                                                          "
"    int lid = get_local_id(0);                                                                             "
"    int group_size = get_local_size(0);                                                                    "
"    partial_sums[lid] = data[get_global_id(0)];                                                            "
"    barrier(CLK_LOCAL_MEM_FENCE);                                                                          "
"    for (int i = group_size / 2; i > 0; i >>= 1) {                                                         "
"        if (lid < i) {                                                                                     "
"            partial_sums[lid] += partial_sums[lid + i];                                                    "
"        }                                                                                                  "
"        barrier(CLK_LOCAL_MEM_FENCE);                                                                      "
"    }                                                                                                      "
"                                                                                                           "
"    if (lid == 0) {                                                                                        "
"        output[get_group_id(0)] = dot(partial_sums[0], (float4)(1.0f));                                    "
"    }                                                                                                      "
"}                                                                                                          "
;


int main(void) {
    // Create the two input vectors
    int i;
    const int LIST_SIZE = 1024;
    size_t local_item_size = 64; // Divide work items into groups of 64

    std::vector<float> A(LIST_SIZE);
    for (i = 0; i < LIST_SIZE; i++) {
        A[i] = i;
    }
    // Load the kernel source code into the array source_str
    const int GROUP_SIZE = LIST_SIZE / 4 / local_item_size;
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
        &device_id, &ret_num_devices);

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        LIST_SIZE * sizeof(float), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        GROUP_SIZE * sizeof(float), NULL, &ret);

    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
        LIST_SIZE * sizeof(float), A.data(), 0, NULL, NULL);
    std::vector<float> zero(GROUP_SIZE, 0.0f);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
        sizeof(float), &zero, 0, NULL, NULL);

    // Create a program from the kernel source
    size_t kernel_size = sizeof(kernel_src);
    cl_program program = clCreateProgramWithSource(context, 1,
        (const char**)&kernel_src, nullptr, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "reduction_vector", &ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, local_item_size * sizeof(float) * 4, nullptr);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&b_mem_obj);

    // Execute the OpenCL kernel on the list
    size_t global_item_size = LIST_SIZE / 4; // Process the entire lists
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
        &global_item_size, &local_item_size, 0, NULL, NULL);

    // Read the memory buffer C on the device to the local variable C
    std::vector<float> res(GROUP_SIZE, 0.0f);
    ret = clEnqueueReadBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
        GROUP_SIZE * sizeof(float), res.data(), 0, NULL, NULL);
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    // Display the result to the screen
    for (size_t i = 0; i < GROUP_SIZE; ++i)
        printf("sum of vector elements: %f\n", res[i]);


    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    return 0;
}