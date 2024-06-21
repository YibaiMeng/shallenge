#ifndef SHA256_CUH_
#define SHA256_CUH_
#include <stdio.h>
#include <stdint.h>

#include <cuda_runtime.h>

#include "sha256_hash.h"


// SHA-256 constants
__device__ __constant__ uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

// Rotate right
__device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n)
{
    return __funnelshift_r(x, x, n);
}

// SHA-256 functions
__device__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t sigma0(uint32_t x)
{
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __forceinline__ uint32_t sigma1(uint32_t x)
{
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ __forceinline__ uint32_t delta0(uint32_t x)
{
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t delta1(uint32_t x)
{
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

__device__ __forceinline__ void sha256_transform(const uint8_t *msg, sha256_hash *hash)
{
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t t1, t2;

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        w[i] = (msg[i * 4] << 24) | (msg[i * 4 + 1] << 16) | (msg[i * 4 + 2] << 8) | msg[i * 4 + 3];
    }
#pragma unroll
    for (int i = 16; i < 64; ++i)
    {
        w[i] = delta1(w[i - 2]) + w[i - 7] + delta0(w[i - 15]) + w[i - 16];
    }
    constexpr uint32_t hash_orig[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

    a = hash_orig[0];
    b = hash_orig[1];
    c = hash_orig[2];
    d = hash_orig[3];
    e = hash_orig[4];
    f = hash_orig[5];
    g = hash_orig[6];
    h = hash_orig[7];

#pragma unroll
    for (int i = 0; i < 64; ++i)
    {
        t1 = h + sigma1(e) + ch(e, f, g) + k[i] + w[i];
        t2 = sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    hash->hash[0] = hash_orig[0] + a;
    hash->hash[1] = hash_orig[1] + b;
    hash->hash[2] = hash_orig[2] + c;
    hash->hash[3] = hash_orig[3] + d;
    hash->hash[4] = hash_orig[4] + e;
    hash->hash[5] = hash_orig[5] + f;
    hash->hash[6] = hash_orig[6] + g;
    hash->hash[7] = hash_orig[7] + h;
}

#endif