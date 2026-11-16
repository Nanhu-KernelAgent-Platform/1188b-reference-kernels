// case17 v2 — moe_permute_with_mask_map（真实形状：T/E/H 运行时传入，R 由 map 推导）
// 设计：block per token；x 行(16KB)一次性入 shared；对该 token 的每个有效 dst
//       全 block 合作拷贝 out[dst] = x[t]（原始字节拷贝，dtype 无关）；
//       permuted_probs[dst] = probs[t,e]。读写均向量化合并。
#include <musa_runtime.h>
#include <stdint.h>

#define THREADS 512

__global__ void permute_rt_kernel(const uint4* __restrict__ x,     // [T, H/8] uint4
                                  const int64_t* __restrict__ map, // [T, E]
                                  const float* __restrict__ probs, // [T, E]
                                  uint4* __restrict__ out,         // [R, H/8]
                                  float* __restrict__ pp,          // [R]
                                  int T, int E, int vec_per_row) {
    extern __shared__ uint4 sx[];          // H 个 bf16 = vec_per_row 个 uint4
    __shared__ int64_t s_ml[1024];         // map 行暂存（E<=1024）
    __shared__ int s_dst[1024];            // 有效 dst 下标（每 token <= E 个）
    __shared__ float s_prob[1024];
    __shared__ int s_cnt;

    int t = blockIdx.x;
    // 1) x[t] -> shared（合并向量化读）
    const uint4* src = x + (size_t)t * vec_per_row;
    for (int v = threadIdx.x; v < vec_per_row; v += THREADS) sx[v] = src[v];

    // 2) map 行合并读入 shared（320 线程并行），再由单线程在 shared 上压实
    for (int e = threadIdx.x; e < E; e += THREADS) s_ml[e] = map[(size_t)t * E + e];
    __syncthreads();
    if (threadIdx.x == 0) {
        int c = 0;
        const float* prow = probs + (size_t)t * E;
        for (int e = 0; e < E; ++e) {
            int64_t d = s_ml[e];
            if (d >= 0) {
                s_dst[c] = (int)d;
                s_prob[c] = prow[e];
                ++c;
            }
        }
        s_cnt = c;
    }
    __syncthreads();

    // 3) 全 block 合作：对每个有效 dst 写出一整行
    int cnt = s_cnt;
    for (int k = 0; k < cnt; ++k) {
        uint4* dstp = out + (size_t)s_dst[k] * vec_per_row;
        for (int v = threadIdx.x; v < vec_per_row; v += THREADS) dstp[v] = sx[v];
        if (threadIdx.x == 0) pp[s_dst[k]] = s_prob[k];
    }
}

void permute_rt_launcher(const void* x, const int64_t* map, const float* probs,
                         void* out, float* pp, int T, int E, int H) {
    int vec_per_row = H * 2 / (int)sizeof(uint4);          // bf16: H*2 字节 / 16B
    size_t shmem = (size_t)vec_per_row * sizeof(uint4);
    permute_rt_kernel<<<T, THREADS, shmem>>>(
        (const uint4*)x, map, probs, (uint4*)out, pp, T, E, vec_per_row);
}
