#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <chrono>

#include <iostream>
#include <iomanip>
#include "sha256.cuh"
#include "cuda_error.cuh"

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

template <int32_t iter_per_kernel, int username_len>
__global__ void find_lowest_sha256(int64_t *run_offset, int64_t *best_nonce, sha256_hash *best_hash)
{
    sha256_hash thread_best_hash;
    memset(reinterpret_cast<void *>(thread_best_hash.hash), 0xff, 32); // Initialize with high values

    char buffer[64] = "yibaimeng/";
    sha256_hash hash;
    int32_t t_id = blockDim.x * blockIdx.x + threadIdx.x;
    int64_t nounce = run_offset[0] + (int64_t)t_id * (int64_t)iter_per_kernel;
    int64_t thread_best_nounce = 0;
    for (int c = 0; c < iter_per_kernel; c++)
    {
        nounce++;

#pragma unroll
        for (int i = 0; i < 64; i += 4)
        {
            buffer[username_len + i / 4] = ('a' + (int)((int64_t)(nounce >> i) & 0xf));
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
}

std::string nounce_to_string(int64_t nounce)
{
    char buffer[20];
    for (int i = 0; i < 64; i += 4)
    {
        buffer[i / 4] = ('a' + (int)((int64_t)(nounce >> i) & 0xf));
    }
    buffer[16] = 0;
    return std::string(buffer);
}

std::string print_hash(const sha256_hash &hash)
{
    char buffer[100];
    snprintf(buffer, sizeof(buffer), "%08x %08x %08x %08x %08x %08x %08x %08x", hash.hash[0], hash.hash[1], hash.hash[2], hash.hash[3], hash.hash[4], hash.hash[5], hash.hash[6], hash.hash[7]);
    return std::string(buffer);
}

constexpr int grid_size = 48;
constexpr int block_size = 1024;
constexpr int64_t num_threads_per_launch = grid_size * block_size;
constexpr int64_t iter_per_thread = 1048576;

int main(int argc, char *argv[0])
{
    int64_t cmd_seed = 0;
    int64_t cmd_iter = 0;

    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--seed" && i + 1 < argc)
        {
            cmd_seed = std::stoll(argv[++i]);
        }
        else if (std::string(argv[i]) == "--iter" && i + 1 < argc)
        {
            cmd_iter = std::stoll(argv[++i]);
        }
    }

    std::cout << "Starting seed: " << cmd_seed << std::endl;
    std::cout << "Iterations: " << cmd_iter << std::endl;
    std::cout << "Threads per launch: " << num_threads_per_launch << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Hashes in total: " << (double)num_threads_per_launch * (double)cmd_iter * (double)iter_per_thread / 1e12 << " TH" << std::endl;
    int64_t next_seed = cmd_seed + (int64_t)num_threads_per_launch * (int64_t)cmd_iter * (int64_t)iter_per_thread;
    if (next_seed < cmd_seed)
    {
        std::cout << "Overflows! no next seed" << std::endl;
    }
    std::cout << "Next seed is " << next_seed << std::endl;
    auto sys_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int64_t *d_best_nonce;
    sha256_hash *d_best_hash;
    int64_t *d_run_offset;
    CUDA_CHECK(cudaMalloc(&d_run_offset, 1 * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_best_nonce, num_threads_per_launch * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_best_hash, num_threads_per_launch * sizeof(sha256_hash)));
    int64_t *h_best_nonce = new int64_t[num_threads_per_launch];
    sha256_hash *h_best_hash = new sha256_hash[num_threads_per_launch];

    sha256_hash program_best_hash;
    memset(reinterpret_cast<void *>(program_best_hash.hash), 0xff, 32); // Initialize with high values

    for (int64_t iter = cmd_seed; iter < cmd_seed + cmd_iter; iter++)
    {
        int64_t nounce_offset = (int64_t)iter * ((int64_t)num_threads_per_launch * (int64_t)iter_per_thread);
        CUDA_CHECK(cudaMemcpy(d_run_offset, &nounce_offset, 1 * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(start, 0));
        find_lowest_sha256<iter_per_thread, 10><<<grid_size, block_size>>>(d_run_offset, d_best_nonce, d_best_hash);
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaMemcpy(h_best_nonce, d_best_nonce, num_threads_per_launch * sizeof(int64_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_best_hash, d_best_hash, num_threads_per_launch * sizeof(sha256_hash), cudaMemcpyDeviceToHost));

        sha256_hash iter_best_hash;
        memset(reinterpret_cast<void *>(iter_best_hash.hash), 0xff, 32); // Initialize with high values
        int64_t iter_best_nounce = -1;
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

        std::cout << std::fixed << std::setprecision(1) << "Hash rate: " << (double)(num_threads_per_launch * iter_per_thread) / (double)elapsedTime / 1e6 << " GH / s" << std::endl;
        if (is_smaller(&iter_best_hash, &program_best_hash))
        {
            std::cout << "Best nonce: " << nounce_to_string(iter_best_nounce) << std::endl;
            std::cout << "Best hash: " << print_hash(iter_best_hash) << std::endl;
            program_best_hash = iter_best_hash;
        }
        std::cout << "Iteration " << iter - cmd_seed + 1 << " of " << cmd_iter << " completed. " << std::endl;
        std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - sys_start).count() / 1000.0 << " s" << std::endl;
    }
    CUDA_CHECK(cudaFree(d_best_nonce));
    CUDA_CHECK(cudaFree(d_best_hash));
    delete[] h_best_hash;
    delete[] h_best_nonce;

    return 0;
}
