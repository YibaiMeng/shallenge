#ifndef CUDA_DEVICE_PROPERTY_CUH_
#define CUDA_DEVICE_PROPERTY_CUH_

#include <iostream>

#include <cuda_runtime.h>

#include "cuda_error.cuh"

void print_cuda_device_property()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    std::cerr << "Number of devices: " << deviceCount << std::endl;

    for (int device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));

        std::cerr << "Device " << device << ": " << deviceProp.name << std::endl;
        std::cerr << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cerr << "  Total global memory: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
        std::cerr << "  Shared memory per block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
        std::cerr << "  Registers per block: " << deviceProp.regsPerBlock << std::endl;
        std::cerr << "  Warp size: " << deviceProp.warpSize << std::endl;
        std::cerr << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cerr << "  Max threads per multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cerr << "  Number of multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
    }
}

template <typename Kernel>
void print_kernel_attributes(Kernel my_kernel)
{

    cudaFuncAttributes funcAttributes;
    CUDA_CHECK(cudaFuncGetAttributes(&funcAttributes, my_kernel));

    int device;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));

    // Print register usage and shared memory usage
    std::cerr << "Register usage: " << funcAttributes.numRegs << std::endl;
    std::cerr << "Shared memory per block: " << funcAttributes.sharedSizeBytes << " bytes" << std::endl;

    // Calculate occupancy
    int maxActiveBlocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks,
        my_kernel,
        funcAttributes.numRegs,
        funcAttributes.sharedSizeBytes));

    int occupancy = maxActiveBlocks * deviceProp.maxThreadsPerMultiProcessor;
    std::cerr << "Occupancy: " << occupancy << " threads per multiprocessor" << std::endl;
}

#endif
