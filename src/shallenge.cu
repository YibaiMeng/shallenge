#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <iomanip>

#include <argparse/argparse.hpp>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "cuda_device_property.cuh"
#include "cuda_error.cuh"
#include "sha256.cuh"
#include "sha256_hash.h"

constexpr int64_t iter_per_kernel = 1048576;
constexpr int block_size = 1024;
#define USERNAME "yibaimeng/RTX3070Ti/mengyibai+dot+com/"
constexpr int username_len = strlen(USERNAME);

__device__ __host__ bool is_smaller(sha256_hash *a, sha256_hash *b)
{
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        uint32_t a_val = a->hash[j];
        uint32_t b_val = b->hash[j];
        if (a_val < b_val)
            return true;
        if (a_val > b_val)
            return false;
    }
    return false;
}

// curand state initialization is done in a different kernel because its quite time consuming when subsequence is larger.
__global__ void init_curand_state(uint64_t seed, curandStateXORWOW_t *state)
{
    uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(seed, tid, 0, state + tid);
}

template <int username_len, int threads_per_block = 1024>
__global__ void find_lowest_sha256(uint64_t *block_best_nonce, sha256_hash *block_best_hash, int32_t iter_per_kernel, curandStateXORWOW_t *curand_state)
{

    int thread_id = threadIdx.x;
    int warp_id = thread_id / 32;
    constexpr int num_warps_per_block = threads_per_block / 32;

    // store warp level reduction
    __shared__ sha256_hash shared_hashes[num_warps_per_block];
    __shared__ uint64_t shared_nonces[num_warps_per_block];

    curandState local_curand_state = curand_state[blockDim.x * blockIdx.x + threadIdx.x];
    sha256_hash thread_best_hash;
    memset(reinterpret_cast<void *>(thread_best_hash.hash), 0xff, 32); // Initialize with high values
    int64_t thread_best_nonce = 0;
    char buffer[64] = USERNAME;
    static_assert(strlen(USERNAME) == username_len);
    static_assert(username_len + 16 < 62); // That's the most a single sha256 block takes?
    sha256_hash hash;
    for (int c = 0; c < iter_per_kernel; c++)
    {
        uint32_t r1 = curand(&local_curand_state);
        uint32_t r2 = curand(&local_curand_state);
        uint64_t nonce = ((uint64_t)r1 << 32) + (uint64_t)r2;

#pragma unroll
        for (int i = 0; i < 64; i += 4)
        {
            buffer[username_len + i / 4] = ('a' + ((nonce >> i) & 0xf));
        }

        constexpr int len = username_len + 16;
        uint8_t *padded_msg = reinterpret_cast<uint8_t *>(buffer);
        padded_msg[len] = 0x80;
        constexpr uint64_t bit_len = len * 8;
        padded_msg[63] = bit_len & 0xff;
        padded_msg[62] = (bit_len >> 8) & 0xff;

        sha256_transform(padded_msg, &hash);
        if (is_smaller(&hash, &thread_best_hash))
        {
            thread_best_hash = hash;
            thread_best_nonce = nonce;
        }
    }

    __syncthreads();

#pragma unroll
    for (int offset = 32 / 2; offset > 0; offset /= 2)
    {
        sha256_hash other_hash;
        uint64_t other_nonce = __shfl_down_sync(0xFFFFFFFF, thread_best_nonce, offset);
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            other_hash.hash[i] = __shfl_down_sync(0xFFFFFFFF, thread_best_hash.hash[i], offset);
        }

        if (is_smaller(&other_hash, &thread_best_hash))
        {
            thread_best_hash = other_hash;
            thread_best_nonce = other_nonce;
        }
    }
    if (thread_id % 32 == 0)
    {
        shared_hashes[warp_id] = thread_best_hash;
        shared_nonces[warp_id] = thread_best_nonce;
    }
    __syncthreads();

    if (thread_id == 0)
    {
#pragma unroll
        for (int i = 0; i < num_warps_per_block; i++)
        {
            if (is_smaller(&shared_hashes[i], &thread_best_hash))
            {
                thread_best_nonce = shared_nonces[i];
                thread_best_hash = shared_hashes[i];
            }
        }
        block_best_nonce[blockIdx.x] = thread_best_nonce;
        block_best_hash[blockIdx.x] = thread_best_hash;
    }

    curand_state[blockDim.x * blockIdx.x + threadIdx.x] = local_curand_state;
}

std::string nonce_to_string(uint64_t nonce)
{
    char buffer[20];
    for (int i = 0; i < 64; i += 4)
    {
        buffer[i / 4] = ('a' + (int)((uint64_t)(nonce >> i) & 0xf));
    }
    buffer[16] = 0;
    return std::string(buffer);
}

int main(int argc, char *argv[0])
{
    argparse::ArgumentParser program("shallenge");

    program.add_argument("--seed")
        .help("set seed value")
        .required()
        .scan<'i', int64_t>();

    program.add_argument("--hashes")
        .help("amount of hashes to compute, in TH.")
        .default_value(1.0)
        .scan<'g', double>();
    program.add_argument("--num_gpus")
        .help("number of GPUs to run the program on")
        .default_value(1)
        .scan<'i', int>();

    program.add_argument("--grid_size")
        .help("grid size for the kernel launch")
        .default_value(48)
        .scan<'i', int>();
    program.add_argument("--dry-run")
        .help("don't run, only show device property and run configurations")
        .default_value(false)
        .implicit_value(true);

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    int num_gpus = program.get<int>("--num_gpus");
    int64_t cmd_seed = program.get<int64_t>("--seed");
    double cmd_hash = program.get<double>("--hashes");
    int grid_size = program.get<int>("--grid_size");
    int64_t num_threads_per_launch = grid_size * block_size;
    double hashes_per_kernel = cmd_hash * 1e12 / (static_cast<double>(num_threads_per_launch) * num_gpus);
    int64_t cmd_iter = static_cast<int64_t>(std::ceil(hashes_per_kernel / iter_per_kernel));

    print_cuda_device_property();
    print_kernel_attributes(find_lowest_sha256<username_len, block_size>);
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count > num_gpus)
    {
        std::cerr << "Number of devices smaller than what's specified in --num_gpus" << std::endl;
        exit(1);
    }

    std::cerr << "Seed: " << cmd_seed << std::endl;
    std::cerr << std::fixed << std::setprecision(3) << "Hashes in total: " << cmd_hash << " TH" << std::endl;
    std::cerr << "Grid size " << grid_size << ", Block size: " << block_size << ", Threads: " << num_threads_per_launch << std::endl;
    std::cerr << "Kernel launches: " << cmd_iter << std::endl;
    if (program.get<bool>("--dry-run"))
    {
        exit(0);
    }

    auto sys_start = std::chrono::high_resolution_clock::now();

    std::vector<cudaEvent_t> start_events(num_gpus), stop_events(num_gpus);
    std::vector<cudaStream_t> streams(num_gpus);
    std::vector<uint64_t *> d_block_best_nonce(num_gpus);
    std::vector<sha256_hash *> d_block_best_hash(num_gpus);
    std::vector<curandStateXORWOW_t *> curand_state(num_gpus);
    std::vector<uint64_t *> h_block_best_nonce(num_gpus);
    std::vector<sha256_hash *> h_block_best_hash(num_gpus);

    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id)
    {
        CUDA_CHECK(cudaSetDevice(gpu_id));
        CUDA_CHECK(cudaStreamCreate(&streams[gpu_id]));
        cudaEvent_t &start = start_events[gpu_id];
        cudaEvent_t &stop = stop_events[gpu_id];
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_block_best_nonce[gpu_id]), grid_size * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_block_best_hash[gpu_id]), grid_size * sizeof(sha256_hash)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&curand_state[gpu_id]), num_threads_per_launch * sizeof(curandStateXORWOW_t)));
        h_block_best_nonce[gpu_id] = new uint64_t[grid_size];
        h_block_best_hash[gpu_id] = new sha256_hash[grid_size];

        init_curand_state<<<grid_size, block_size, 0, streams[gpu_id]>>>(cmd_seed + gpu_id, curand_state[gpu_id]);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    sha256_hash program_best_hash;
    memset(reinterpret_cast<void *>(program_best_hash.hash), 0xff, 32); // Initialize with high values
    for (int64_t iter = 0; iter < cmd_iter; iter++)
    {
        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id)
        {
            CUDA_CHECK(cudaSetDevice(gpu_id));
            CUDA_CHECK(cudaEventRecord(start_events[gpu_id], streams[gpu_id]));
            find_lowest_sha256<username_len, block_size><<<grid_size, block_size, 0, streams[gpu_id]>>>(d_block_best_nonce[gpu_id], d_block_best_hash[gpu_id], iter_per_kernel, curand_state[gpu_id]);
            CUDA_CHECK(cudaEventRecord(stop_events[gpu_id], streams[gpu_id]));
        }

        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id)
        {
            CUDA_CHECK(cudaSetDevice(gpu_id));
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(h_block_best_nonce[gpu_id]), reinterpret_cast<void *>(d_block_best_nonce[gpu_id]), grid_size * sizeof(uint64_t), cudaMemcpyDeviceToHost, streams[gpu_id]));
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(h_block_best_hash[gpu_id]), reinterpret_cast<void *>(d_block_best_hash[gpu_id]), grid_size * sizeof(sha256_hash), cudaMemcpyDeviceToHost, streams[gpu_id]));
        }

        sha256_hash iter_best_hash;
        memset(reinterpret_cast<void *>(iter_best_hash.hash), 0xff, 32); // Initialize with high values
        uint64_t iter_best_nonce = 0;
        float total_elasped_time = 0;
        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id)
        {
            CUDA_CHECK(cudaSetDevice(gpu_id));
            CUDA_CHECK(cudaStreamSynchronize(streams[gpu_id]));
            CUDA_CHECK(cudaEventSynchronize(stop_events[gpu_id]));
            float elapsed_time;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_events[gpu_id], stop_events[gpu_id]));
            total_elasped_time += elapsed_time;
            for (int i = 0; i < grid_size; i++)
            {
                if (is_smaller(h_block_best_hash[gpu_id] + i, &iter_best_hash))
                {
                    iter_best_hash = h_block_best_hash[gpu_id][i];
                    iter_best_nonce = h_block_best_nonce[gpu_id][i];
                }
            }
        }

        std::cerr << std::fixed << std::setprecision(2) << "Hash rate per GPU: " << (double)(num_threads_per_launch * iter_per_kernel * num_gpus) / (double)total_elasped_time / 1e6 << " GH / s" << std::endl;
        if (is_smaller(&iter_best_hash, &program_best_hash))
        {
            std::cerr << "Best nonce: " << nonce_to_string(iter_best_nonce) << std::endl;
            std::cerr << "Best hash: " << iter_best_hash << std::endl;
            program_best_hash = iter_best_hash;
        }
        std::cerr << "Iteration " << iter + 1 << " of " << cmd_iter << " completed. " << std::endl;
        std::cerr << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - sys_start).count() / 1000.0 << " s" << std::endl;
    }

    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id)
    {
        CUDA_CHECK(cudaSetDevice(gpu_id));
        CUDA_CHECK(cudaFree(d_block_best_nonce[gpu_id]));
        CUDA_CHECK(cudaFree(d_block_best_hash[gpu_id]));
        delete[] h_block_best_hash[gpu_id];
        delete[] h_block_best_nonce[gpu_id];
        CUDA_CHECK(cudaStreamDestroy(streams[gpu_id]));
    }

    return 0;
}
