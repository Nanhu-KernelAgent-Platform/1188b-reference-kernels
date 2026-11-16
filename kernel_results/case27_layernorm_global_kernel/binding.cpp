#include <torch/extension.h>
#include <ATen/ATen.h>

extern "C" void launch_rmsnorm(const void* x, const void* gamma, float* y_acc, void* y, float* rstd, int M, float eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rmsnorm_forward(torch::Tensor x, torch::Tensor gamma, double eps) {
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16, "gamma must be bf16");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    int64_t M = x.size(0);
    int64_t N = x.size(1);
    TORCH_CHECK(N == 8192, "only N=8192 supported");
    TORCH_CHECK(gamma.size(0) == N, "gamma size mismatch");

    auto y_acc = torch::empty({M, N}, x.options().dtype(torch::kFloat32));
    auto y = torch::empty({M, N}, x.options());
    auto rstd = torch::empty({M}, x.options().dtype(torch::kFloat32));

    launch_rmsnorm(x.data_ptr(), gamma.data_ptr(), y_acc.data_ptr<float>(), y.data_ptr(), rstd.data_ptr<float>(), (int)M, (float)eps);

    return std::make_tuple(y_acc, y, rstd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward");
}
