#include <musa_runtime.h>
#include <stdint.h>

namespace {
constexpr int HEADS = 128;
constexpr int K_DIM = 128;
constexpr int V_DIM = 128;
constexpr int EMB_DIM = 64;
constexpr int HALF = EMB_DIM / 2;
constexpr int DK_LAST = K_DIM + EMB_DIM;   // 192  (k content + rope)
constexpr int DKV_LAST = K_DIM + V_DIM;    // 256

__device__ __forceinline__ float bf_lo(uint32_t x) {
    union { uint32_t u; float f; } v;
    v.u = (x & 0x0000ffffu) << 16;
    return v.f;
}
__device__ __forceinline__ float bf_hi(uint32_t x) {
    union { uint32_t u; float f; } v;
    v.u = x & 0xffff0000u;
    return v.f;
}
__device__ __forceinline__ float bf2f(uint16_t x) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)x << 16;
    return v.f;
}
__device__ __forceinline__ uint16_t f2bf(float f) {
    union { float f; uint32_t u; } v;
    v.f = f;
    uint32_t b = v.u;
    uint32_t lsb = (b >> 16) & 1u;
    b += 0x7fffu + lsb;
    return (uint16_t)(b >> 16);
}
}  // namespace

template <int AW> struct VecN;
template <> struct alignas(16) VecN<16> { uint32_t w[4]; };
template <> struct alignas(32) VecN<32> { uint32_t w[8]; };

// CACHE 0: default caching.  CACHE 1: streaming loads/stores (__ldcs/__stcs).
template <int CACHE, int AW> struct Mem;

template <> struct Mem<0, 16> {
    static __device__ __forceinline__ VecN<16> ld(const VecN<16>* p) { return *p; }
    static __device__ __forceinline__ void st(VecN<16>* p, const VecN<16>& v) { *p = v; }
};
template <> struct Mem<1, 16> {
    static __device__ __forceinline__ VecN<16> ld(const VecN<16>* p) {
        const int4 t = __ldcs(reinterpret_cast<const int4*>(p));
        VecN<16> r;
        r.w[0] = (uint32_t)t.x; r.w[1] = (uint32_t)t.y;
        r.w[2] = (uint32_t)t.z; r.w[3] = (uint32_t)t.w;
        return r;
    }
    static __device__ __forceinline__ void st(VecN<16>* p, const VecN<16>& v) {
        int4 t;
        t.x = (int)v.w[0]; t.y = (int)v.w[1]; t.z = (int)v.w[2]; t.w = (int)v.w[3];
        __stcs(reinterpret_cast<int4*>(p), t);
    }
};
template <> struct Mem<0, 32> {
    static __device__ __forceinline__ VecN<32> ld(const VecN<32>* p) { return *p; }
    static __device__ __forceinline__ void st(VecN<32>* p, const VecN<32>& v) { *p = v; }
};
template <> struct Mem<1, 32> {
    static __device__ __forceinline__ VecN<32> ld(const VecN<32>* p) {
        const int4* q = reinterpret_cast<const int4*>(p);
        const int4 a = __ldcs(q);
        const int4 b = __ldcs(q + 1);
        VecN<32> r;
        r.w[0] = (uint32_t)a.x; r.w[1] = (uint32_t)a.y;
        r.w[2] = (uint32_t)a.z; r.w[3] = (uint32_t)a.w;
        r.w[4] = (uint32_t)b.x; r.w[5] = (uint32_t)b.y;
        r.w[6] = (uint32_t)b.z; r.w[7] = (uint32_t)b.w;
        return r;
    }
    static __device__ __forceinline__ void st(VecN<32>* p, const VecN<32>& v) {
        int4 a, b;
        a.x = (int)v.w[0]; a.y = (int)v.w[1]; a.z = (int)v.w[2]; a.w = (int)v.w[3];
        b.x = (int)v.w[4]; b.y = (int)v.w[5]; b.z = (int)v.w[6]; b.w = (int)v.w[7];
        int4* q = reinterpret_cast<int4*>(p);
        __stcs(q, a);
        __stcs(q + 1, b);
    }
};

// MODE 0: single interleaved copy loop (k content + v content in one pass)
// MODE 1: separate k pass and v pass
template <int NT, int MODE, int AW, int CACHE>
__global__ __launch_bounds__(NT) void rotary_bwd_kv_kernel(
    const uint16_t* __restrict__ dk,
    const uint16_t* __restrict__ dv,
    const uint16_t* __restrict__ gcos,
    const uint16_t* __restrict__ gsin,
    uint16_t* __restrict__ d_kv,
    uint16_t* __restrict__ d_emb,
    int rows, int batch) {
    constexpr int EPT = AW / 2;
    constexpr int ROPEVECS = (EMB_DIM * 2) / AW;
    constexpr int NPART = NT / ROPEVECS;
    constexpr int DK_VECS = (DK_LAST * 2) / AW;
    constexpr int K_VECS = (K_DIM * 2) / AW;
    constexpr int V_VECS = (V_DIM * 2) / AW;
    constexpr int DKV_VECS = (DKV_LAST * 2) / AW;
    constexpr int TOTAL_KV = HEADS * DKV_VECS;
    typedef Mem<CACHE, AW> M;

    extern __shared__ float sred[];

    const int tid = threadIdx.x;
    const int slot = tid / ROPEVECS;
    const int voff = tid % ROPEVECS;

    for (int r = blockIdx.x; r < rows; r += gridDim.x) {
        __syncthreads();

        const int64_t dk_base = (int64_t)r * (int64_t)HEADS * DK_LAST;
        const int64_t dv_base = (int64_t)r * (int64_t)HEADS * V_DIM;
        const int64_t kv_base = (int64_t)r * (int64_t)HEADS * DKV_LAST;

        const VecN<AW>* __restrict__ sk = reinterpret_cast<const VecN<AW>*>(dk + dk_base);
        const VecN<AW>* __restrict__ sv = reinterpret_cast<const VecN<AW>*>(dv + dv_base);
        VecN<AW>* __restrict__ dst = reinterpret_cast<VecN<AW>*>(d_kv + kv_base);

        if (MODE == 0) {
#pragma unroll
            for (int i = tid; i < TOTAL_KV; i += NT) {
                const int h = i / DKV_VECS;
                const int part = i % DKV_VECS;
                if (part < K_VECS) {
                    M::st(dst + i, M::ld(sk + h * DK_VECS + part));
                } else {
                    M::st(dst + i, M::ld(sv + h * V_VECS + (part - K_VECS)));
                }
            }
        } else {
#pragma unroll
            for (int i = tid; i < HEADS * K_VECS; i += NT) {
                const int h = i / K_VECS;
                const int part = i % K_VECS;
                M::st(dst + h * DKV_VECS + part, M::ld(sk + h * DK_VECS + part));
            }
#pragma unroll
            for (int i = tid; i < HEADS * V_VECS; i += NT) {
                const int h = i / V_VECS;
                const int part = i % V_VECS;
                M::st(dst + h * DKV_VECS + K_VECS + part, M::ld(sv + h * V_VECS + part));
            }
        }

        const VecN<AW>* __restrict__ rope =
            reinterpret_cast<const VecN<AW>*>(dk + dk_base + K_DIM);
        float acc[EPT];
#pragma unroll
        for (int j = 0; j < EPT; ++j) acc[j] = 0.0f;
#pragma unroll 4
        for (int h = slot; h < HEADS; h += NPART) {
            const VecN<AW> t = M::ld(rope + h * DK_VECS + voff);
#pragma unroll
            for (int j = 0; j < EPT; j += 2) {
                acc[j] += bf_lo(t.w[j / 2]);
                acc[j + 1] += bf_hi(t.w[j / 2]);
            }
        }
#pragma unroll
        for (int j = 0; j < EPT; ++j) sred[slot * EMB_DIM + voff * EPT + j] = acc[j];

        __syncthreads();

        if (tid < HALF) {
            float L = 0.0f, R = 0.0f;
#pragma unroll
            for (int s = 0; s < NPART; ++s) {
                L += sred[s * EMB_DIM + tid];
                R += sred[s * EMB_DIM + HALF + tid];
            }
            const int64_t e = ((int64_t)(r / batch)) * EMB_DIM;
            const float c_L = bf2f(gcos[e + tid]);
            const float c_R = bf2f(gcos[e + HALF + tid]);
            const float s_L = bf2f(gsin[e + tid]);
            const float s_R = bf2f(gsin[e + HALF + tid]);
            uint16_t* out = d_emb + (int64_t)r * EMB_DIM;
            out[2 * tid] = f2bf(L * c_L + R * s_R);
            out[2 * tid + 1] = f2bf(R * c_R - L * s_L);
        }
    }
}

// Roofline probe: identical memory traffic, no rope reduction / d_emb write.
template <int NT, int AW, int CACHE>
__global__ __launch_bounds__(NT) void probe_copy_kernel(
    const uint16_t* __restrict__ dk,
    const uint16_t* __restrict__ dv,
    uint16_t* __restrict__ d_kv,
    int rows) {
    constexpr int DK_VECS = (DK_LAST * 2) / AW;
    constexpr int K_VECS = (K_DIM * 2) / AW;
    constexpr int V_VECS = (V_DIM * 2) / AW;
    constexpr int DKV_VECS = (DKV_LAST * 2) / AW;
    constexpr int TOTAL_KV = HEADS * DKV_VECS;
    typedef Mem<CACHE, AW> M;

    const int tid = threadIdx.x;
    for (int r = blockIdx.x; r < rows; r += gridDim.x) {
        const int64_t dk_base = (int64_t)r * (int64_t)HEADS * DK_LAST;
        const int64_t dv_base = (int64_t)r * (int64_t)HEADS * V_DIM;
        const int64_t kv_base = (int64_t)r * (int64_t)HEADS * DKV_LAST;
        const VecN<AW>* __restrict__ sk = reinterpret_cast<const VecN<AW>*>(dk + dk_base);
        const VecN<AW>* __restrict__ sv = reinterpret_cast<const VecN<AW>*>(dv + dv_base);
        VecN<AW>* __restrict__ dst = reinterpret_cast<VecN<AW>*>(d_kv + kv_base);
#pragma unroll
        for (int i = tid; i < TOTAL_KV; i += NT) {
            const int h = i / DKV_VECS;
            const int part = i % DKV_VECS;
            if (part < K_VECS) {
                M::st(dst + i, M::ld(sk + h * DK_VECS + part));
            } else {
                M::st(dst + i, M::ld(sv + h * V_VECS + (part - K_VECS)));
            }
        }
    }
}

template <int NT, int MODE, int AW, int CACHE>
static void run(const void* dk, const void* dv, const void* cos, const void* sin,
                void* d_kv, void* d_emb, int64_t rows, int64_t batch, int rows_per_block) {
    constexpr int ROPEVECS = (EMB_DIM * 2) / AW;
    constexpr int NPART = NT / ROPEVECS;
    const size_t shmem = (size_t)NPART * EMB_DIM * sizeof(float);
    int64_t blocks = (rows + rows_per_block - 1) / rows_per_block;
    if (blocks < 1) blocks = 1;
    rotary_bwd_kv_kernel<NT, MODE, AW, CACHE><<<(unsigned)blocks, NT, shmem>>>(
        reinterpret_cast<const uint16_t*>(dk),
        reinterpret_cast<const uint16_t*>(dv),
        reinterpret_cast<const uint16_t*>(cos),
        reinterpret_cast<const uint16_t*>(sin),
        reinterpret_cast<uint16_t*>(d_kv),
        reinterpret_cast<uint16_t*>(d_emb),
        (int)rows, (int)batch);
}

template <int NT, int AW, int CACHE>
static void run_probe(const void* dk, const void* dv, void* d_kv, int64_t rows) {
    probe_copy_kernel<NT, AW, CACHE><<<(unsigned)rows, NT>>>(
        reinterpret_cast<const uint16_t*>(dk),
        reinterpret_cast<const uint16_t*>(dv),
        reinterpret_cast<uint16_t*>(d_kv),
        (int)rows);
}

#define LAUNCH(nt, mode, aw, cache)                                              \
    run<nt, mode, aw, cache>(dk, dv, cos, sin, d_kv, d_emb, rows, batch, rows_per_block)

#define DISPATCH_CACHE(nt, mode, aw)                         \
    do {                                                     \
        if (cache == 0) { LAUNCH(nt, mode, aw, 0); return; }  \
        LAUNCH(nt, mode, aw, 1);                              \
        return;                                              \
    } while (0)

void launch_rotary_bwd_kv(const void* dk, const void* dv, const void* cos, const void* sin,
                          void* d_kv, void* d_emb, int64_t rows, int64_t batch,
                          int nt, int mode, int aw, int rows_per_block, int cache) {
    if (mode == 0) {
        if (aw == 16) {
            if (nt == 128) DISPATCH_CACHE(128, 0, 16);
            if (nt == 512) DISPATCH_CACHE(512, 0, 16);
            DISPATCH_CACHE(256, 0, 16);
        }
        if (nt == 128) DISPATCH_CACHE(128, 0, 32);
        if (nt == 512) DISPATCH_CACHE(512, 0, 32);
        DISPATCH_CACHE(256, 0, 32);
    }
    if (aw == 16) {
        if (nt == 128) DISPATCH_CACHE(128, 1, 16);
        if (nt == 512) DISPATCH_CACHE(512, 1, 16);
        DISPATCH_CACHE(256, 1, 16);
    }
    if (nt == 128) DISPATCH_CACHE(128, 1, 32);
    if (nt == 512) DISPATCH_CACHE(512, 1, 32);
    DISPATCH_CACHE(256, 1, 32);
}

void launch_probe(const void* dk, const void* dv, void* d_kv, int64_t rows,
                  int nt, int aw, int cache) {
#define PROBE(nt_, aw_, c_)                                     \
    do {                                                        \
        if (nt == nt_ && aw == aw_ && cache == c_) {            \
            run_probe<nt_, aw_, c_>(dk, dv, d_kv, rows);        \
            return;                                             \
        }                                                       \
    } while (0)
    PROBE(128, 16, 0); PROBE(128, 16, 1);
    PROBE(256, 16, 0); PROBE(256, 16, 1);
    PROBE(512, 16, 0); PROBE(512, 16, 1);
    PROBE(128, 32, 0); PROBE(128, 32, 1);
    PROBE(256, 32, 0); PROBE(256, 32, 1);
    PROBE(512, 32, 0); PROBE(512, 32, 1);
#undef PROBE
}
