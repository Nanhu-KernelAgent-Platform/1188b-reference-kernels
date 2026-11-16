#include <musa_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float bf16_to_float(uint16_t x) {
    union { uint32_t u; float f; } v;
    v.u = static_cast<uint32_t>(x) << 16;
    return v.f;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void delta_kernel_128(const uint16_t* __restrict__ out,
                                 const uint16_t* __restrict__ dout,
                                 float* __restrict__ delta,
                                 int S, int H, int D) {
    __shared__ float smem[128];

    int h = blockIdx.y;
    int s0 = blockIdx.x * 128;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    #pragma unroll
    for (int iter = 0; iter < 4; ++iter) {
        int s = s0 + warp + iter * 32;
        if (s < S) {
            const uint16_t* po = out + (static_cast<size_t>(s) * H + h) * D;
            const uint16_t* pd = dout + (static_cast<size_t>(s) * H + h) * D;
            int base = lane * 4;
            unsigned long long vo =
                *reinterpret_cast<const unsigned long long*>(po + base);
            unsigned long long vd =
                *reinterpret_cast<const unsigned long long*>(pd + base);

            float sum = 0.f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint16_t a = static_cast<uint16_t>((vo >> (16 * i)) & 0xffff);
                uint16_t b = static_cast<uint16_t>((vd >> (16 * i)) & 0xffff);
                sum += bf16_to_float(a) * bf16_to_float(b);
            }
            sum = warp_reduce_sum(sum);
            if (lane == 0) smem[warp + iter * 32] = sum;
        } else {
            if (lane == 0) smem[warp + iter * 32] = 0.f;
        }
    }

    __syncthreads();

    int tid = threadIdx.x;
    if (tid < 128) {
        int s = s0 + tid;
        if (s < S) {
            delta[h * static_cast<size_t>(S) + s] = smem[tid];
        }
    }
}

__global__ void delta_kernel_generic(const uint16_t* __restrict__ out,
                                     const uint16_t* __restrict__ dout,
                                     float* __restrict__ delta,
                                     int S, int H, int D) {
    int h = blockIdx.y;
    int s = blockIdx.x * blockDim.y + threadIdx.y;
    if (s >= S) return;

    const uint16_t* po = out + (static_cast<size_t>(s) * H + h) * D;
    const uint16_t* pd = dout + (static_cast<size_t>(s) * H + h) * D;

    float sum = 0.f;
    for (int d = threadIdx.x; d < D; d += 32) {
        sum += bf16_to_float(po[d]) * bf16_to_float(pd[d]);
    }
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) {
        delta[h * static_cast<size_t>(S) + s] = sum;
    }
}

void launch_delta_bf16(const uint16_t* out, const uint16_t* dout,
                       float* delta, int S, int H, int D) {
    if (D == 128) {
        dim3 block(1024);
        dim3 grid((S + 127) / 128, H);
        delta_kernel_128<<<grid, block>>>(out, dout, delta, S, H, D);
    } else {
        const int warps_per_block = 8;
        dim3 block(32, warps_per_block);
        dim3 grid((S + warps_per_block - 1) / warps_per_block, H);
        delta_kernel_generic<<<grid, block>>>(out, dout, delta, S, H, D);
    }
}
