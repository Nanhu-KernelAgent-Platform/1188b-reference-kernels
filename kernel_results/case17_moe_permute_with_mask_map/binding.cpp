// case17 v2 binding — T/E/H 运行时推导；R = max(map)+1 实时归约，无任何硬编码
#include <torch/extension.h>
#include <ATen/ATen.h>

void permute_rt_launcher(const void* x, const int64_t* map, const float* probs,
                         void* out, float* pp, int T, int E, int H);

std::tuple<at::Tensor, at::Tensor> permute_mask_map(const at::Tensor& x,
                                                    const at::Tensor& row_id_map,
                                                    const at::Tensor& probs) {
    TORCH_CHECK(x.dim() == 2, "x must be [T,H]");
    TORCH_CHECK(row_id_map.dim() == 2, "row_id_map must be [T,E]");
    TORCH_CHECK(row_id_map.size(0) == x.size(0), "map/probs token dim mismatch");
    const int T = (int)x.size(0);
    const int H = (int)x.size(1);
    const int E = (int)row_id_map.size(1);
    TORCH_CHECK(E <= 1024, "E exceeds shared compaction capacity");

    auto xc = x.contiguous();
    auto mc = row_id_map.contiguous();
    auto pc = probs.contiguous().to(at::kFloat);

    // R = max(row_id_map) + 1（设备归约 + 同步取值）
    int64_t R = mc.max().item<int64_t>() + 1;

    auto out = torch::empty({R, H}, xc.options());                 // 全部被覆盖，无需零填
    auto pp = torch::empty({R}, pc.options().dtype(at::kFloat));

    permute_rt_launcher(xc.data_ptr(), mc.data_ptr<int64_t>(), pc.data_ptr<float>(),
                        out.data_ptr(), pp.data_ptr<float>(), T, E, H);
    return std::make_tuple(out, pp);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("permute_mask_map", &permute_mask_map, "permute with mask map (true-shape generic)");
}
