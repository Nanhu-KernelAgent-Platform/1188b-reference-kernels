#include <musa_runtime.h>

__device__ __forceinline__ unsigned short fp32_to_bf16_rne_bits(float x) {
    union { float f; unsigned int u; } v;
    v.f = x;
    unsigned int lsb = (v.u >> 16) & 1u;
    unsigned int rounded = v.u + 0x7FFFu + lsb;
    return (unsigned short)(rounded >> 16);
}

__global__ void cast_fp32_to_bf16_rne_kernel(const float* __restrict__ src,
                                             unsigned short* __restrict__ dst,
                                             long long n) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(src + base);
        unsigned short h0 = fp32_to_bf16_rne_bits(v.x);
        unsigned short h1 = fp32_to_bf16_rne_bits(v.y);
        unsigned short h2 = fp32_to_bf16_rne_bits(v.z);
        unsigned short h3 = fp32_to_bf16_rne_bits(v.w);
        uint2 p;
        p.x = (unsigned int)h0 | ((unsigned int)h1 << 16);
        p.y = (unsigned int)h2 | ((unsigned int)h3 << 16);
        *reinterpret_cast<uint2*>(dst + base) = p;
    } else {
        for (long long i = base; i < n; ++i) {
            dst[i] = fp32_to_bf16_rne_bits(src[i]);
        }
    }
}

void launch_cast_fp32_to_bf16_rne(const float* src, unsigned short* dst, long long n) {
    if (n <= 0) return;
    const int block = 1024;
    long long threads4 = (n + 3) / 4;
    int grid = (int)((threads4 + block - 1) / block);
    cast_fp32_to_bf16_rne_kernel<<<grid, block>>>(src, dst, n);
}
