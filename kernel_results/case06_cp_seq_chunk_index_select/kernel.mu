#include <musa_runtime.h>
#include <stdint.h>

__global__ void index_select_bf16_vec_kernel(const uint16_t* __restrict__ src,
                                             uint16_t* __restrict__ dst,
                                             const int32_t* __restrict__ index,
                                             int64_t inner_vec) {
    const int64_t row = blockIdx.y;
    const int64_t src_row = static_cast<int64_t>(index[row]);
    const int64_t base = inner_vec * 8;
    const uint4* s = reinterpret_cast<const uint4*>(src + src_row * base);
    uint4* d = reinterpret_cast<uint4*>(dst + row * base);
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < inner_vec) {
        d[i] = s[i];
    }
}

__global__ void index_select_bf16_scalar_kernel(const uint16_t* __restrict__ src,
                                                uint16_t* __restrict__ dst,
                                                const int32_t* __restrict__ index,
                                                int64_t inner) {
    const int64_t row = blockIdx.y;
    const int64_t src_row = static_cast<int64_t>(index[row]);
    const uint16_t* s = src + src_row * inner;
    uint16_t* d = dst + row * inner;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < inner;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        d[i] = s[i];
    }
}

void launch_index_select_bf16_vec(const uint16_t* src, uint16_t* dst,
                                  const int32_t* index, int64_t out_rows,
                                  int64_t inner_vec) {
    dim3 block(256);
    dim3 grid(static_cast<unsigned int>((inner_vec + 255) / 256),
              static_cast<unsigned int>(out_rows));
    index_select_bf16_vec_kernel<<<grid, block>>>(src, dst, index, inner_vec);
}

void launch_index_select_bf16_scalar(const uint16_t* src, uint16_t* dst,
                                     const int32_t* index, int64_t out_rows,
                                     int64_t inner) {
    dim3 block(256);
    dim3 grid(static_cast<unsigned int>((inner + 255) / 256),
              static_cast<unsigned int>(out_rows));
    index_select_bf16_scalar_kernel<<<grid, block>>>(src, dst, index, inner);
}
