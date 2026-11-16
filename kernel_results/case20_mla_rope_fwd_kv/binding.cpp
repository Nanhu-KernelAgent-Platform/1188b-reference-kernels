#include <torch/extension.h>
#include <ATen/ATen.h>
#include <tuple>

void launch_mla_rope_kv(const void* kv, const void* k_pos,
                        const float* cos, const float* sin,
                        void* key, int S);

std::tuple<torch::Tensor, torch::Tensor> mla_rope_kv(
    torch::Tensor kv, torch::Tensor k_pos_emb,
    torch::Tensor cos, torch::Tensor sin) {
    TORCH_CHECK(kv.scalar_type() == torch::kBFloat16, "kv must be bf16");
    TORCH_CHECK(k_pos_emb.scalar_type() == torch::kBFloat16, "k_pos_emb must be bf16");
    TORCH_CHECK(cos.scalar_type() == torch::kFloat32, "cos must be float32");
    TORCH_CHECK(sin.scalar_type() == torch::kFloat32, "sin must be float32");
    TORCH_CHECK(kv.is_contiguous(), "kv must be contiguous");
    TORCH_CHECK(k_pos_emb.is_contiguous(), "k_pos_emb must be contiguous");
    TORCH_CHECK(cos.is_contiguous(), "cos must be contiguous");
    TORCH_CHECK(sin.is_contiguous(), "sin must be contiguous");

    int64_t S = kv.size(0);
    int64_t B = kv.size(1);
    int64_t H = kv.size(2);
    int64_t D = kv.size(3);
    int64_t E = k_pos_emb.size(3);

    TORCH_CHECK(B == 1, "B must be 1");
    TORCH_CHECK(H == 128, "H must be 128");
    TORCH_CHECK(D == 256, "kv last dim must be 256");
    TORCH_CHECK(E == 64, "E must be 64");
    TORCH_CHECK(cos.size(0) == S && cos.size(3) == E, "cos shape mismatch");
    TORCH_CHECK(sin.size(0) == S && sin.size(3) == E, "sin shape mismatch");
    TORCH_CHECK(k_pos_emb.size(0) == S && k_pos_emb.size(1) == B,
                "k_pos_emb shape mismatch");

    auto key = torch::empty({S, B, H, 128 + E}, kv.options());
    auto v = kv.narrow(3, 128, 128);

    launch_mla_rope_kv(kv.data_ptr(), k_pos_emb.data_ptr(),
                       cos.data_ptr<float>(), sin.data_ptr<float>(),
                       key.data_ptr(), static_cast<int>(S));

    return std::make_tuple(key, v);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mla_rope_kv", &mla_rope_kv, "MLA RoPE KV forward");
}
