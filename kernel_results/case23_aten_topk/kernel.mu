#include <musa_runtime.h>
#include <stdint.h>

#ifndef ROWS_PER_BLOCK
#define ROWS_PER_BLOCK 16
#endif
#define LANES 8                  /* lanes handling one row (4 rows per warp) */
#define MAX_K 8
#define NEG_INF (-3.402823466e+38F)
#define IDX_INF 0x7FFFFFFF

// Row-wise top-k (largest, sorted, stable) for fp32 [M, N] router scores.
// One lane group of LANES threads owns one row; each lane keeps its own
// sorted top-8 in registers, then the row is reduced with a warp-synchronous
// pairwise merge over shared memory in SoA layout (bank-conflict free).
__global__ void topk_kernel(const float* __restrict__ input,
                            float* __restrict__ out_values,
                            int64_t* __restrict__ out_indices,
                            int M, int N, int K) {
    const int grp  = threadIdx.x / LANES;
    const int lane = threadIdx.x % LANES;
    const int row  = blockIdx.x * ROWS_PER_BLOCK + grp;
    if (row >= M) return;

    __shared__ float s_val[ROWS_PER_BLOCK * MAX_K * LANES];
    __shared__ int   s_idx[ROWS_PER_BLOCK * MAX_K * LANES];

    float* sv = s_val + grp * (MAX_K * LANES);
    int*   si = s_idx + grp * (MAX_K * LANES);

    float lv[MAX_K];
    int li[MAX_K];
    int cnt = 0;
#pragma unroll
    for (int j = 0; j < MAX_K; ++j) { lv[j] = NEG_INF; li[j] = IDX_INF; }

    const float* row_ptr = input + (int64_t)row * N;
    for (int i = lane; i < N; i += LANES) {
        const float v = row_ptr[i];
        if (v > lv[MAX_K - 1]) {              /* fast reject against current 8th */
            int pos = 0;
            while (pos < MAX_K - 1 && lv[pos] >= v) ++pos;
            const int end = (cnt < MAX_K) ? cnt : MAX_K - 1;
#pragma unroll
            for (int j = MAX_K - 1; j > 0; --j) {
                if (j <= pos || j > end) continue;
                lv[j] = lv[j - 1];
                li[j] = li[j - 1];
            }
            lv[pos] = v; li[pos] = i;
            if (cnt < MAX_K) ++cnt;
        }
    }

#pragma unroll
    for (int j = 0; j < MAX_K; ++j) { sv[j * LANES + lane] = lv[j]; si[j * LANES + lane] = li[j]; }
    __syncwarp();

    for (int stride = LANES / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            const int r = lane + stride;
            float a[MAX_K], b[MAX_K];
            int ia[MAX_K], ib[MAX_K];
#pragma unroll
            for (int j = 0; j < MAX_K; ++j) {
                a[j] = sv[j * LANES + lane]; ia[j] = si[j * LANES + lane];
                b[j] = sv[j * LANES + r];    ib[j] = si[j * LANES + r];
            }
            int pa = 0, pb = 0;
#pragma unroll
            for (int c = 0; c < MAX_K; ++c) {
                /* stable order: larger value first, equal value -> smaller index */
                const bool takeA = (a[pa] > b[pb]) || (a[pa] == b[pb] && ia[pa] < ib[pb]);
                float ov; int oi;
                if (takeA) { ov = a[pa]; oi = ia[pa]; pa = (pa < MAX_K - 1) ? pa + 1 : MAX_K - 1; }
                else       { ov = b[pb]; oi = ib[pb]; pb = (pb < MAX_K - 1) ? pb + 1 : MAX_K - 1; }
                sv[c * LANES + lane] = ov;
                si[c * LANES + lane] = oi;
            }
        }
        __syncwarp();
    }

    if (lane == 0) {
        for (int c = 0; c < K; ++c) {
            out_values[(int64_t)row * K + c] = sv[c * LANES];
            out_indices[(int64_t)row * K + c] = (int64_t)si[c * LANES];
        }
    }
}

void launch_topk(const float* input, float* out_values, int64_t* out_indices,
                 int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    const int blocks = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    topk_kernel<<<blocks, ROWS_PER_BLOCK * LANES>>>(input, out_values, out_indices, M, N, K);
}
