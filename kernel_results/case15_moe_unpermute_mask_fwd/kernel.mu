#include <musa_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float bf16_to_float(uint16_t v) {
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)v) << 16;
    return c.f;
}

__device__ __forceinline__ uint16_t float_to_bf16(float f) {
    union { float f; uint32_t u; } c;
    c.f = f;
    uint32_t u = c.u;
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

__global__ void moe_unpermute_kernel(const uint16_t* __restrict__ pin,
                                     const int64_t* __restrict__ row_id_map,
                                     uint16_t* __restrict__ out,
                                     int T, int H, int E) {
    int t = blockIdx.x;
    if (t >= T) return;

    __shared__ int smem_src[512];
    __shared__ int smem_count;
    int tid = threadIdx.x;

    if (tid == 0) smem_count = 0;
    __syncthreads();

    for (int e = tid; e < E; e += blockDim.x) {
        int64_t src = row_id_map[(int64_t)e * T + t];
        if (src >= 0) {
            int pos = atomicAdd(&smem_count, 1);
            if (pos < 512) smem_src[pos] = (int)src;
        }
    }
    __syncthreads();

    int cnt = smem_count;
    if (cnt > 512) cnt = 512;

    int vec_per_row = H >> 3;
    for (int v = tid; v < vec_per_row; v += blockDim.x) {
        float acc[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = 0.0f;

        for (int k = 0; k < cnt; ++k) {
            int src = smem_src[k];
            const uint4* row = reinterpret_cast<const uint4*>(pin + (int64_t)src * H);
            uint4 val = row[v];
            acc[0] += bf16_to_float((uint16_t)(val.x & 0xFFFFu));
            acc[1] += bf16_to_float((uint16_t)(val.x >> 16));
            acc[2] += bf16_to_float((uint16_t)(val.y & 0xFFFFu));
            acc[3] += bf16_to_float((uint16_t)(val.y >> 16));
            acc[4] += bf16_to_float((uint16_t)(val.z & 0xFFFFu));
            acc[5] += bf16_to_float((uint16_t)(val.z >> 16));
            acc[6] += bf16_to_float((uint16_t)(val.w & 0xFFFFu));
            acc[7] += bf16_to_float((uint16_t)(val.w >> 16));
        }

        uint16_t b0 = float_to_bf16(acc[0]);
        uint16_t b1 = float_to_bf16(acc[1]);
        uint16_t b2 = float_to_bf16(acc[2]);
        uint16_t b3 = float_to_bf16(acc[3]);
        uint16_t b4 = float_to_bf16(acc[4]);
        uint16_t b5 = float_to_bf16(acc[5]);
        uint16_t b6 = float_to_bf16(acc[6]);
        uint16_t b7 = float_to_bf16(acc[7]);

        uint4 o;
        o.x = ((uint32_t)b1 << 16) | b0;
        o.y = ((uint32_t)b3 << 16) | b2;
        o.z = ((uint32_t)b5 << 16) | b4;
        o.w = ((uint32_t)b7 << 16) | b6;

        uint4* out_row = reinterpret_cast<uint4*>(out + (int64_t)t * H);
        out_row[v] = o;
    }

    int rem_start = (H >> 3) << 3;
    for (int c = rem_start + tid; c < H; c += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < cnt; ++k) {
            int src = smem_src[k];
            acc += bf16_to_float(pin[(int64_t)src * H + c]);
        }
        out[(int64_t)t * H + c] = float_to_bf16(acc);
    }
}

void launch_moe_unpermute(const uint16_t* pin,
                          const int64_t* row_id_map,
                          uint16_t* out,
                          int T, int H, int E) {
    dim3 grid(T);
    dim3 block(256);
    moe_unpermute_kernel<<<grid, block>>>(pin, row_id_map, out, T, H, E);
}
