#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <cstdint>

void launch_index_select_bf16_vec(const uint16_t* src, uint16_t* dst,
                                  const int32_t* index, int64_t out_rows,
                                  int64_t inner_vec);
void launch_index_select_bf16_scalar(const uint16_t* src, uint16_t* dst,
                                     const int32_t* index, int64_t out_rows,
                                     int64_t inner);

at::Tensor index_select_bf16(const at::Tensor& x, const at::Tensor& index) {
    TORCH_CHECK(x.is_musa(), "x must be a MUSA tensor");
    TORCH_CHECK(index.is_musa(), "index must be a MUSA tensor");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(index.scalar_type() == at::kInt, "index must be int32");
    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
    TORCH_CHECK(index.dim() == 1, "index must be one-dimensional");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(index.is_contiguous(), "index must be contiguous");

    const int64_t out_rows = index.numel();
    int64_t inner = 1;
    for (int64_t d = 1; d < x.dim(); ++d) {
        inner *= x.size(d);
    }

    std::vector<int64_t> out_shape;
    out_shape.reserve(x.dim());
    out_shape.push_back(out_rows);
    for (int64_t d = 1; d < x.dim(); ++d) {
        out_shape.push_back(x.size(d));
    }

    at::Tensor out = at::empty(out_shape, x.options());
    if (out_rows == 0 || inner == 0) {
        return out;
    }

    const uint16_t* src = reinterpret_cast<const uint16_t*>(x.data_ptr());
    uint16_t* dst = reinterpret_cast<uint16_t*>(out.data_ptr());
    const int32_t* idx = reinterpret_cast<const int32_t*>(index.data_ptr());

    if ((inner & 7) == 0) {
        launch_index_select_bf16_vec(src, dst, idx, out_rows, inner / 8);
    } else {
        launch_index_select_bf16_scalar(src, dst, idx, out_rows, inner);
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("index_select_bf16", &index_select_bf16, "index_select dim0 bf16");
}
