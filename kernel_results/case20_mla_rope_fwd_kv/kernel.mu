#include <musa_runtime.h>
#include <stdint.h>

constexpr int K_DIM = 128;
constexpr int V_DIM = 128;
constexpr int EMB = 64;
constexpr int NHEADS = 128;

__device__ __forceinline__ float bf16_to_float(uint16_t x) {
    union { uint32_t u; float f; } p;
    p.u = static_cast<uint32_t>(x) << 16;
    return p.f;
}

__device__ __forceinline__ uint16_t float_to_bf16(float x) {
    union { uint32_t u; float f; } p;
    p.f = x;
    uint32_t u = p.u;
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

__global__ void mla_rope_kv_kernel(const uint16_t* __restrict__ kv,
                                   const uint16_t* __restrict__ k_pos,
                                   const float* __restrict__ cos,
                                   const float* __restrict__ sin,
                                   uint16_t* __restrict__ key) {
    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ uint4 rope_vec[8];
    uint16_t* rope = reinterpret_cast<uint16_t*>(rope_vec);

    if (tid < EMB) {
        const int half = EMB / 2;
        const uint16_t* kp = k_pos + s * EMB;
        const float* c = cos + s * EMB;
        const float* sn = sin + s * EMB;
        float val;
        if (tid < half) {
            int j = tid;
            float x1 = bf16_to_float(kp[2 * j]);
            float x2 = bf16_to_float(kp[2 * j + 1]);
            volatile float t1 = x1 * c[j];
            volatile float t2 = x2 * sn[j];
            val = t1 - t2;
        } else {
            int j = tid - half;
            float x1 = bf16_to_float(kp[2 * j]);
            float x2 = bf16_to_float(kp[2 * j + 1]);
            volatile float t1 = x2 * c[half + j];
            volatile float t2 = x1 * sn[half + j];
            val = t1 + t2;
        }
        rope[tid] = float_to_bf16(val);
    }
    __syncthreads();

    const uint4* kv4 = reinterpret_cast<const uint4*>(kv + s * NHEADS * (K_DIM + V_DIM));
    uint4* key4 = reinterpret_cast<uint4*>(key + s * NHEADS * (K_DIM + EMB));

    constexpr int K_VEC = K_DIM / 8;
    constexpr int R_VEC = EMB / 8;
    constexpr int KEY_STRIDE = (K_DIM + EMB) / 8;
    constexpr int KV_STRIDE = (K_DIM + V_DIM) / 8;

    for (int t = tid; t < NHEADS * K_VEC; t += blockDim.x) {
        int h = t / K_VEC;
        int v = t % K_VEC;
        key4[h * KEY_STRIDE + v] = kv4[h * KV_STRIDE + v];
    }

    for (int t = tid; t < NHEADS * R_VEC; t += blockDim.x) {
        int h = t / R_VEC;
        int v = t % R_VEC;
        key4[h * KEY_STRIDE + K_VEC + v] = rope_vec[v];
    }
}

void launch_mla_rope_kv(const void* kv, const void* k_pos,
                        const float* cos, const float* sin,
                        void* key, int S) {
    mla_rope_kv_kernel<<<S, 256>>>(
        reinterpret_cast<const uint16_t*>(kv),
        reinterpret_cast<const uint16_t*>(k_pos),
        cos, sin,
        reinterpret_cast<uint16_t*>(key));
}
