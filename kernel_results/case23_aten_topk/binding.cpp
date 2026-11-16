#include <torch/extension.h>
#include <ATen/ATen.h>

void launch_topk(const float* input, float* out_values, int64_t* out_indices, int M, int N, int K);

std::tuple<torch::Tensor, torch::Tensor> topk_forward(torch::Tensor input, int64_t k) {
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "only float32 is supported");
    TORCH_CHECK(k >= 1 && k <= input.size(1), "k out of range");
    TORCH_CHECK(k <= 8, "k must be <= 8");

    auto x = input.contiguous();
    const int64_t M = x.size(0);
    const int64_t N = x.size(1);
    auto values = torch::empty({M, k}, x.options());
    auto indices = torch::empty({M, k}, x.options().dtype(at::kLong));

    if (M > 0 && N > 0) {
        launch_topk(x.data_ptr<float>(), values.data_ptr<float>(),
                    indices.data_ptr<int64_t>(), static_cast<int>(M),
                    static_cast<int>(N), static_cast<int>(k));
    }
    return std::make_tuple(values, indices);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_forward", &topk_forward, "row-wise top-k forward (largest, sorted)");
}
