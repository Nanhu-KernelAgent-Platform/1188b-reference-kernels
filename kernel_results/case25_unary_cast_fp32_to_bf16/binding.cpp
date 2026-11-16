#include <torch/extension.h>
#include <ATen/ATen.h>

void launch_cast_fp32_to_bf16_rne(const float* src, unsigned short* dst, long long n);

at::Tensor cast_fp32_to_bf16_rne(const at::Tensor& inp) {
    TORCH_CHECK(inp.scalar_type() == at::kFloat, "expected float32 input");
    TORCH_CHECK(inp.is_contiguous(), "expected contiguous input");
    auto out = at::empty(inp.sizes(), inp.options().dtype(at::kBFloat16));
    const float* src = inp.data_ptr<float>();
    unsigned short* dst = reinterpret_cast<unsigned short*>(out.data_ptr());
    launch_cast_fp32_to_bf16_rne(src, dst, static_cast<long long>(inp.numel()));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cast_fp32_to_bf16_rne", &cast_fp32_to_bf16_rne, "fp32 to bf16 RNE cast");
}
