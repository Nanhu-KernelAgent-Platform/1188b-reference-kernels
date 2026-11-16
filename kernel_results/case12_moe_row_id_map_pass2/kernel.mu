// case12 v2 — row_id_map 构建（真实形状：T 运行时传入，E 任意支持）
// 设计：输入转置 [T,E]->[E,T]（合并访存）=> 按 expert 行计数 => 1-block 前缀和
//       => 按 expert 行 coalesced 填充 row_t [E,T] => 64 位转置 row_t->row [T,E]
#include <musa_runtime.h>
#include <stdint.h>

#define SCAN_BLOCK 1024

// ---- K0: bool 转置 [T,E] -> [E,T]（32x32 tile，合并读写）----
__global__ void transpose_bool_kernel(const unsigned char* in, unsigned char* out, int T, int E) {
    __shared__ unsigned char tile[32][33];
    int bx = blockIdx.x * 32;  // token 维（列）
    int by = blockIdx.y * 32;  // expert 维（行）
    int tx = threadIdx.x, ty = threadIdx.y;
    // 读 in[T,E]（行主序）：固定行 i 时 32 线程沿 e 连续 -> 合并
    for (int i = ty; i < 32; i += blockDim.y) {
        int t = bx + i;
        int e = by + tx;
        unsigned char v = 0;
        if (t < T && e < E) v = in[(size_t)t * E + e];
        tile[i][tx] = v;
    }
    __syncthreads();
    for (int i = ty; i < 32; i += blockDim.y) {
        int e = by + i;
        int t = bx + tx;
        if (e < E && t < T) out[(size_t)e * T + t] = tile[tx][i];
    }
}

// ---- K1: 每个 expert 的 token 计数（block per expert，行连续合并读）----
__global__ void count_kernel(const unsigned char* inT, int T, int* counts) {
    int j = blockIdx.x;
    const unsigned char* row = inT + (size_t)j * T;
    __shared__ int part[SCAN_BLOCK];
    int local = 0;
    for (int t = threadIdx.x; t < T; t += blockDim.x) local += row[t] ? 1 : 0;
    part[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) counts[j] = part[0];
}

// ---- K2: counts 前缀和（1 block，串行 320 次加法即可）----
__global__ void offsets_kernel(const int* counts, int* offsets, int E) {
    if (threadIdx.x == 0) {
        int run = 0;
        for (int k = 0; k < E; ++k) { offsets[k] = run; run += counts[k]; }
    }
}

// ---- K3: 填充 row_t [E,T]（block per expert = 1 warp，warp 内 ballot/popc 秩计算，零 __syncthreads）----
__global__ void fill_warp_kernel(const unsigned char* inT, int T, const int* offsets, int64_t* row_t) {
    int j = blockIdx.x;
    const unsigned char* src = inT + (size_t)j * T;
    int64_t* dst = row_t + (size_t)j * T;
    int base = offsets[j];
    int lane = threadIdx.x;                    // blockDim.x = 32
    for (int t0 = 0; t0 < T; t0 += 32) {
        int t = t0 + lane;
        bool m = (t < T && src[t]);
        unsigned b = __ballot_sync(0xffffffffu, m ? 1 : 0);
        int rank = __popc(b & ((1u << lane) - 1u));
        int cnt = __popc(b);
        if (t < T) dst[t] = m ? (int64_t)(base + rank) : (int64_t)-1;
        base += cnt;                           // bposc 全 warp 一致，断序推进
    }
}

// ---- K4: int64 转置 [E,T] -> [T,E]（32x32 tile，合并读写）----
__global__ void transpose_i64_kernel(const int64_t* in, int64_t* out, int E, int T) {
    __shared__ int64_t tile[32][33];
    int bx = blockIdx.x * 32;  // t 维
    int by = blockIdx.y * 32;  // e 维
    int tx = threadIdx.x, ty = threadIdx.y;
    for (int i = ty; i < 32; i += blockDim.y) {
        int e = by + i;
        int t = bx + tx;
        int64_t v = -1;
        if (e < E && t < T) v = in[(size_t)e * T + t];
        tile[i][tx] = v;
    }
    __syncthreads();
    for (int i = ty; i < 32; i += blockDim.y) {
        int t = bx + i;
        int e = by + tx;
        if (t < T && e < E) out[(size_t)t * E + e] = tile[tx][i];
    }
}

// ---- launchers ----
void transpose_bool_launcher(const unsigned char* in, unsigned char* out, int T, int E) {
    dim3 grid((T + 31) / 32, (E + 31) / 32);
    transpose_bool_kernel<<<grid, dim3(32, 8)>>>(in, out, T, E);
}
void count_launcher(const unsigned char* inT, int T, int E, int* counts) {
    count_kernel<<<E, SCAN_BLOCK>>>(inT, T, counts);
}
void offsets_launcher(const int* counts, int* offsets, int E) {
    offsets_kernel<<<1, 32>>>(counts, offsets, E);
}
void fill_launcher_rt(const unsigned char* inT, int T, const int* offsets, int64_t* row_t, int E) {
    fill_warp_kernel<<<E, 32>>>(inT, T, offsets, row_t);
}
void transpose_i64_launcher(const int64_t* in, int64_t* out, int E, int T) {
    dim3 grid((T + 31) / 32, (E + 31) / 32);
    transpose_i64_kernel<<<grid, dim3(32, 8)>>>(in, out, E, T);
}
