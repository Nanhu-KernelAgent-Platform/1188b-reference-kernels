#include <musa_runtime.h>
#include <musa_bf16.h>

struct alignas(16) bf16x8 { __mt_bfloat16 x[8]; };

template <int N, int THREADS, int VEC>
__global__ void rmsnorm_kernel(const __mt_bfloat16* __restrict__ x,
                               const __mt_bfloat16* __restrict__ gamma,
                               float* __restrict__ y_acc,
                               __mt_bfloat16* __restrict__ y,
                               float* __restrict__ rstd,
                               float eps) {
    constexpr int VEC_PER_ROW = N / VEC;
    constexpr int ITEMS = VEC_PER_ROW / THREADS;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int row_offset = row * N;

    const bf16x8* xv = reinterpret_cast<const bf16x8*>(x + row_offset);
    const bf16x8* gv = reinterpret_cast<const bf16x8*>(gamma);
    bf16x8* yv = reinterpret_cast<bf16x8*>(y + row_offset);
    float4* y_acc4 = reinterpret_cast<float4*>(y_acc + row_offset);

    float vals[ITEMS][VEC];
    float sum = 0.0f;
    for (int i = 0; i < ITEMS; ++i) {
        int vidx = tid + i * THREADS;
        bf16x8 v = xv[vidx];
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            float f = __bfloat162float(v.x[j]);
            vals[i][j] = f;
            sum += f * f;
        }
    }

    __shared__ float sdata[THREADS];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    __syncthreads();
    float inv = rsqrtf(sdata[0] / (float)N + eps);
    if (tid == 0) rstd[row] = inv;

    for (int i = 0; i < ITEMS; ++i) {
        int vidx = tid + i * THREADS;
        bf16x8 g = gv[vidx];
        bf16x8 out_bf;
        float out_f[VEC];
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            float g_f = __bfloat162float(g.x[j]);
            float o = vals[i][j] * inv * g_f;
            out_f[j] = o;
            out_bf.x[j] = __float2bfloat16(o);
        }
        yv[vidx] = out_bf;
        float4* dst = &y_acc4[vidx * 2];
        dst[0] = make_float4(out_f[0], out_f[1], out_f[2], out_f[3]);
        dst[1] = make_float4(out_f[4], out_f[5], out_f[6], out_f[7]);
    }
}

extern "C" void launch_rmsnorm(const void* x, const void* gamma, float* y_acc, void* y, float* rstd, int M, float eps) {
    rmsnorm_kernel<8192, 256, 8><<<M, 256>>>(
        reinterpret_cast<const __mt_bfloat16*>(x),
        reinterpret_cast<const __mt_bfloat16*>(gamma),
        y_acc,
        reinterpret_cast<__mt_bfloat16*>(y),
        rstd,
        eps);
}
