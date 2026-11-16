"""case15（真实形状版）— MoE unpermute (combine 前向): moe_unpermute_mask<bf16,float,long,false,false,true,8>

形状来源（唯一权威）: 021-1188B-real-ops-11-11-record.xlsx
  trace: x1w_1188B_128k_ac-full_rank-8.json（seq 131072 / CP 8 / 每卡 16384 tokens）
  记录: pin(R,8192) + row_id_map(320,16384) -> out(16384,8192)，R=80,820-99,179/层
  profiler 真实平均: 3.27 ms
本版构造: 与 case12 同一套确定性 routing_map（R=86,016）；pin(R,8192) bf16。
注意: 旧 kernel_cases 版用 TOKENS=4096/TOPK=8（R=32768），形状作废。
"""

import torch
import torch.nn as nn

# ---- real per-rank dims (xlsx record, 128k seq / CP 8) ----
TOKENS = 16384
HID = 8192
NUM_EXPERTS = 320

CASE = {
    "id": 15,
    "name": "moe_unpermute_mask_fwd",
    "kernel": "moe_unpermute_mask<__mt_bfloat16, float, long, false, false, true, 8>",
    "csv_row": 15,
    "profiler_avg_ms": 3.27,
    "shape_provenance": "021-1188B-real-ops-11-11-record.xlsx (x1w_1188B_128k_ac-full_rank-8.json)",
    "backward": False,
}


def make_routing_map():
    torch.manual_seed(20261112)
    sel = torch.argsort(torch.rand(TOKENS, NUM_EXPERTS), dim=-1)[:, :6]
    rm = torch.zeros(TOKENS, NUM_EXPERTS, dtype=torch.bool)
    rm.scatter_(1, sel[:, :5], True)
    extra = (torch.arange(TOKENS) % 4 == 1)
    extra_rows = extra.nonzero(as_tuple=True)[0]
    rm[extra_rows] = rm[extra_rows].scatter(1, sel[extra_rows, 5:6], True)  # 行对齐写入
    assert int(rm.sum()) == 86016
    return rm                                                  # R = rm.sum() = 86,016


def make_row_id_map_trans(routing_map):
    """routing_map [T,E] -> row_id_map [E,T] int64（trans，-1 = 未选中）。"""
    mask = routing_map.to(torch.int64)
    within = torch.cumsum(mask, dim=0) - 1
    counts = mask.sum(dim=0)
    offset = torch.cumsum(counts, dim=0) - counts
    row = torch.where(mask > 0, within + offset.unsqueeze(0), torch.full_like(within, -1))
    return row.t().contiguous()


class Model(nn.Module):
    """out[t] = Σ_e pin[row_id_map[e,t]]（只累加被选中的 expert；fp32 累加再回 bf16）。"""

    def __init__(self, num_experts=NUM_EXPERTS):
        super().__init__()

    def forward(self, pin: torch.Tensor, row_id_map: torch.Tensor) -> torch.Tensor:
        valid = row_id_map >= 0
        src = row_id_map[valid]
        dst = valid.nonzero(as_tuple=True)[1]
        out = torch.zeros(row_id_map.shape[1], pin.shape[1], dtype=torch.float32, device=pin.device)
        out.index_add_(0, dst, pin.index_select(0, src).float())
        return out.to(pin.dtype)


def get_init_inputs():
    return [NUM_EXPERTS]


def get_inputs():
    rm = make_routing_map()
    row_id_map = make_row_id_map_trans(rm)
    r_rows = int(rm.sum())
    torch.manual_seed(20261113)
    pin = torch.randn(r_rows, HID).to(torch.bfloat16)
    return [pin, row_id_map]
