#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cstdint>

void launch_moe_unpermute(const uint16_t* pin,
                          const int64_t* row_id_map,
                          uint16_t* out,
                          int T, int H, int E);

torch::Tensor moe_unpermute(torch::Tensor pin, torch::Tensor row_id_map) {
    int E = row_id_map.size(0);
    int T = row_id_map.size(1);
    int H = pin.size(1);
    auto out = torch::empty({T, H}, pin.options());
    launch_moe_unpermute(reinterpret_cast<const uint16_t*>(pin.data_ptr()),
                         row_id_map.data_ptr<int64_t>(),
                         reinterpret_cast<uint16_t*>(out.data_ptr()),
                         T, H, E);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_unpermute", &moe_unpermute, "moe unpermute mask forward");
}
