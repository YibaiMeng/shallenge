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

template <int username_len>
__global__ void find_lowest_sha256(uint64_t *best_nonce, sha256_hash *best_hash, int32_t iter_per_kernel, curandStateXORWOW_t *curand_state)
{
    sha256_hash thread_best_hash;
    memset(reinterpret_cast<void *>(thread_best_hash.hash), 0xff, 32); // Initialize with high values
    static_assert(strlen(USERNAME) == username_len);
    static_assert(username_len + 16 < 62); // That's the most a single sha256 block takes?

    char buffer[64] = USERNAME;
    sha256_hash hash;
    int32_t t_id = blockDim.x * blockIdx.x + threadIdx.x;
    int64_t thread_best_nounce = 0;

    curandState local_curand_state = curand_state[t_id];

    for (int c = 0; c < iter_per_kernel; c++)
    {
        uint32_t r1 = curand(&local_curand_state);
        uint32_t r2 = curand(&local_curand_state);
        uint64_t nounce = ((uint64_t)r1 << 32) + (uint64_t)r2;

#pragma unroll
        for (int i = 0; i < 64; i += 4)
        {
            buffer[username_len + i / 4] = ('a' + ((nounce >> i) & 0xf));
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
            thread_best_nounce = nounce;
        }
    }

    __syncthreads();

    best_nonce[t_id] = thread_best_nounce;
    best_hash[t_id] = thread_best_hash;
    curand_state[t_id] = local_curand_state;
}

std::string nounce_to_string(uint64_t nounce)
{
    char buffer[20];
    for (int i = 0; i < 64; i += 4)
    {
        buffer[i / 4] = ('a' + (int)((uint64_t)(nounce >> i) & 0xf));
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
    program.add_argument("--grid_size")
        .help("grid size for the kernel launch")
        .default_value(48)
        .scan<'i', int>();
    program.add_argument("--block_size")
        .help("block size for the kernel launch")
        .default_value(1024)
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

    int64_t cmd_seed = program.get<int64_t>("--seed");
    double cmd_hash = program.get<double>("--hashes");
    int grid_size = program.get<int>("--grid_size");
    int block_size = program.get<int>("--block_size");
    int64_t num_threads_per_launch = grid_size * block_size;
    double hashes_per_kernel = cmd_hash * 1e12 / static_cast<double>(num_threads_per_launch);
    int64_t cmd_iter = static_cast<int64_t>(std::ceil(hashes_per_kernel / iter_per_kernel));

    print_cuda_device_property();
    print_kernel_attributes(find_lowest_sha256<username_len>);

    std::cerr << "Seed: " << cmd_seed << std::endl;
    std::cerr << std::fixed << std::setprecision(3) << "Hashes in total: " << cmd_hash << " TH" << std::endl;
    std::cerr << "Grid size " << grid_size << ", Block size: " << block_size << ", Threads: " << num_threads_per_launch << std::endl;
    std::cerr << "Kernel launches: " << cmd_iter << std::endl;
    if (program.get<bool>("--dry-run"))
    {
        exit(0);
    }

    auto sys_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    uint64_t *d_best_nonce;
    sha256_hash *d_best_hash;
    curandStateXORWOW_t *curand_state;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_best_nonce), num_threads_per_launch * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_best_hash), num_threads_per_launch * sizeof(sha256_hash)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&curand_state), num_threads_per_launch * sizeof(curandStateXORWOW_t)));
    uint64_t *h_best_nonce = new uint64_t[num_threads_per_launch];
    sha256_hash *h_best_hash = new sha256_hash[num_threads_per_launch];

    sha256_hash program_best_hash;
    memset(reinterpret_cast<void *>(program_best_hash.hash), 0xff, 32); // Initialize with high values

    init_curand_state<<<grid_size, block_size>>>(cmd_seed, curand_state);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int64_t iter = 0; iter < cmd_iter; iter++)
    {
        CUDA_CHECK(cudaEventRecord(start, 0));
        find_lowest_sha256<username_len><<<grid_size, block_size>>>(d_best_nonce, d_best_hash, iter_per_kernel, curand_state);
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(h_best_nonce), reinterpret_cast<void *>(d_best_nonce), num_threads_per_launch * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(h_best_hash), reinterpret_cast<void *>(d_best_hash), num_threads_per_launch * sizeof(sha256_hash), cudaMemcpyDeviceToHost));

        sha256_hash iter_best_hash;
        memset(reinterpret_cast<void *>(iter_best_hash.hash), 0xff, 32); // Initialize with high values
        uint64_t iter_best_nounce = 0;
        for (int i = 0; i < num_threads_per_launch; i++)
        {
            if (is_smaller(h_best_hash + i, &iter_best_hash))
            {
                iter_best_hash = h_best_hash[i];
                iter_best_nounce = h_best_nonce[i];
            }
        }

        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsedTime;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

        std::cerr << std::fixed << std::setprecision(2) << "Hash rate: " << (double)(num_threads_per_launch * iter_per_kernel) / (double)elapsedTime / 1e6 << " GH / s" << std::endl;
        if (is_smaller(&iter_best_hash, &program_best_hash))
        {
            std::cerr << "Best nonce: " << nounce_to_string(iter_best_nounce) << std::endl;
            std::cerr << "Best hash: " << iter_best_hash << std::endl;
            program_best_hash = iter_best_hash;
        }
        std::cerr << "Iteration " << iter + 1 << " of " << cmd_iter << " completed. " << std::endl;
        std::cerr << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - sys_start).count() / 1000.0 << " s" << std::endl;
    }
    CUDA_CHECK(cudaFree(d_best_nonce));
    CUDA_CHECK(cudaFree(d_best_hash));
    delete[] h_best_hash;
    delete[] h_best_nonce;

    return 0;
}
