// case12 v2 binding — 形状全部运行时推导（T=input.size(0), E=input.size(1)），无任何硬编码
#include <torch/extension.h>
#include <ATen/ATen.h>

void transpose_bool_launcher(const unsigned char* in, unsigned char* out, int T, int E);
void count_launcher(const unsigned char* inT, int T, int E, int* counts);
void offsets_launcher(const int* counts, int* offsets, int E);
void fill_launcher_rt(const unsigned char* inT, int T, const int* offsets, int64_t* row_t, int E);
void transpose_i64_launcher(const int64_t* in, int64_t* out, int E, int T);

std::tuple<at::Tensor, at::Tensor> row_id_map(const at::Tensor& input) {
    TORCH_CHECK(input.dim() == 2, "routing_map must be [T,E]");
    TORCH_CHECK(input.dtype() == at::kBool || input.dtype() == at::kByte,
                "routing_map must be bool/byte");
    const int T = (int)input.size(0);
    const int E = (int)input.size(1);

    auto bopt = input.options().dtype(at::kByte);
    auto input_b = input.contiguous().to(at::kByte);          // 保证字节缓冲
    auto inT = torch::empty({E, T}, bopt);                    // 转置缓冲
    auto counts = torch::empty({E}, input.options().dtype(at::kInt));
    auto offsets = torch::empty({E}, input.options().dtype(at::kInt));
    auto row_t = torch::empty({E, T}, input.options().dtype(at::kLong));
    auto row = torch::empty({T, E}, input.options().dtype(at::kLong));

    transpose_bool_launcher(input_b.data_ptr<unsigned char>(),
                            inT.data_ptr<unsigned char>(), T, E);
    count_launcher(inT.data_ptr<unsigned char>(), T, E, counts.data_ptr<int>());
    offsets_launcher(counts.data_ptr<int>(), offsets.data_ptr<int>(), E);
    fill_launcher_rt(inT.data_ptr<unsigned char>(), T, offsets.data_ptr<int>(),
                     row_t.data_ptr<int64_t>(), E);
    transpose_i64_launcher(row_t.data_ptr<int64_t>(), row.data_ptr<int64_t>(), E, T);
    return std::make_tuple(row_t, row);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("row_id_map", &row_id_map, "row_id_map (true-shape generic)");
}
