#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

void launch_rotary_bwd_kv(const void* dk, const void* dv, const void* cos, const void* sin,
                          void* d_kv, void* d_emb, int64_t rows, int64_t batch,
                          int nt, int mode, int aw, int rows_per_block, int cache);
void launch_probe(const void* dk, const void* dv, void* d_kv, int64_t rows,
                  int nt, int aw, int cache);

static int g_nt = 512;
static int g_mode = 1;
static int g_aw = 16;
static int g_rows = 1;
static int g_cache = 0;

void set_config(int nt, int mode, int aw, int rows_per_block, int cache) {
    g_nt = nt;
    g_mode = mode;
    g_aw = aw;
    g_rows = rows_per_block;
    g_cache = cache;
}

torch::Tensor probe_copy(torch::Tensor dk, torch::Tensor dv, int nt, int aw, int cache) {
    const int64_t rows = dk.size(0) * dk.size(1);
    auto out = torch::empty({dk.size(0), dk.size(1), 128, 256}, dk.options());
    launch_probe(dk.data_ptr(), dv.data_ptr(), out.data_ptr(), rows, nt, aw, cache);
    return out;
}

std::vector<torch::Tensor> rotary_bwd_kv(torch::Tensor dk, torch::Tensor dv,
                                        torch::Tensor cos, torch::Tensor sin) {
    TORCH_CHECK(dk.scalar_type() == at::kBFloat16 && dv.scalar_type() == at::kBFloat16 &&
                    cos.scalar_type() == at::kBFloat16 && sin.scalar_type() == at::kBFloat16,
                "all inputs must be bf16");
    TORCH_CHECK(dk.dim() == 4 && dv.dim() == 4, "dk/dv must be 4D");
    TORCH_CHECK(dk.size(2) == 128 && dk.size(3) == 192, "dk must be [seq,batch,128,192]");
    TORCH_CHECK(dv.size(2) == 128 && dv.size(3) == 128, "dv must be [seq,batch,128,128]");

    auto dkc = dk.contiguous();
    auto dvc = dv.contiguous();
    auto cosc = cos.contiguous();
    auto sinc = sin.contiguous();

    const int64_t seq = dkc.size(0);
    const int64_t batch = dkc.size(1);
    const int64_t rows = seq * batch;

    auto d_kv = torch::empty({seq, batch, 128, 256}, dkc.options());
    auto d_emb = torch::empty({seq, batch, 1, 64}, dkc.options());

    launch_rotary_bwd_kv(dkc.data_ptr(), dvc.data_ptr(), cosc.data_ptr(), sinc.data_ptr(),
                         d_kv.data_ptr(), d_emb.data_ptr(), rows, batch,
                         g_nt, g_mode, g_aw, g_rows, g_cache);
    return {d_kv, d_emb};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotary_bwd_kv", &rotary_bwd_kv, "MLA YARN rotary backward (KV)");
    m.def("set_config", &set_config, "tuning knobs: nt, mode, aw, rows_per_block, cache");
    m.def("probe_copy", &probe_copy, "roofline probe: same traffic, no rope math");
}
