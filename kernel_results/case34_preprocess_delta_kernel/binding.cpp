#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cstdint>

void launch_delta_bf16(const uint16_t* out, const uint16_t* dout,
                       float* delta, int S, int H, int D);

torch::Tensor preprocess_delta(torch::Tensor out, torch::Tensor dout) {
    TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bfloat16");
    TORCH_CHECK(dout.scalar_type() == at::kBFloat16, "dout must be bfloat16");
    TORCH_CHECK(out.dim() == 3, "out must be 3D");
    TORCH_CHECK(dout.dim() == 3, "dout must be 3D");
    TORCH_CHECK(out.sizes().equals(dout.sizes()), "out and dout must have same shape");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(dout.is_contiguous(), "dout must be contiguous");

    int64_t S = out.size(0);
    int64_t H = out.size(1);
    int64_t D = out.size(2);

    auto delta = torch::empty({H, S}, out.options().dtype(at::kFloat));

    const uint16_t* out_ptr =
        reinterpret_cast<const uint16_t*>(out.data_ptr<at::BFloat16>());
    const uint16_t* dout_ptr =
        reinterpret_cast<const uint16_t*>(dout.data_ptr<at::BFloat16>());
    float* delta_ptr = delta.data_ptr<float>();

    launch_delta_bf16(out_ptr, dout_ptr, delta_ptr,
                      static_cast<int>(S), static_cast<int>(H), static_cast<int>(D));
    return delta;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("preprocess_delta", &preprocess_delta, "preprocess delta");
}
