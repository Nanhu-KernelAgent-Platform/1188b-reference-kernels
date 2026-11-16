"""case12（真实形状版）— MoE row_id_map 构建（第 2 遍）: _row_id_map_pass_2_kernel

形状来源（唯一权威）: 021-1188B-real-ops-11-11-record.xlsx
  trace: x1w_1188B_128k_ac-full_rank-8.json（seq 131072 / CP 8 / 每卡 16384 tokens）
  记录: routing_map(16384,320) bool -> row_id_map(320,16384) int64 + token_map(16384,320) int64
        每层有效行数 R=80,820-99,179（topk 8 中被 capacity/group 限制后实际路由数）
  profiler 真实平均: 4.24 ms
本版构造: 每 token 选中 5 或 6 个 expert（t%4==1 -> 6），ΣR = 86,016 ∈ [80820, 99179]，seed 固定。
注意: 旧 kernel_cases 版用 TOKENS=4096 且强制 topk=8（R=32768），形状作废。
"""

import torch
import torch.nn as nn

# ---- real per-rank dims (xlsx record, 128k seq / CP 8) ----
TOKENS = 16384   # 每卡 tokens
NUM_EXPERTS = 320

CASE = {
    "id": 12,
    "name": "moe_row_id_map_pass2",
    "kernel": "_row_id_map_pass_2_kernel",
    "csv_row": 12,
    "profiler_avg_ms": 4.24,
    "shape_provenance": "021-1188B-real-ops-11-11-record.xlsx (x1w_1188B_128k_ac-full_rank-8.json)",
    "backward": False,
}


def make_routing_map():
    """确定性构造 routing_map [T,E] bool，R = 86,016（xlsx 记录区间 [80820, 99179] 内）。"""
    torch.manual_seed(20261112)
    sel = torch.argsort(torch.rand(TOKENS, NUM_EXPERTS), dim=-1)[:, :6]   # 候选 6 个
    rm = torch.zeros(TOKENS, NUM_EXPERTS, dtype=torch.bool)
    rm.scatter_(1, sel[:, :5], True)                                       # 基础每 token 5 个
    extra = (torch.arange(TOKENS) % 4 == 1)                                # 1/4 token 加第 6 个
    extra_rows = extra.nonzero(as_tuple=True)[0]
    rm[extra_rows] = rm[extra_rows].scatter(1, sel[extra_rows, 5:6], True)  # 行对齐写入
    assert int(rm.sum()) == 5 * TOKENS + int(extra.sum())                  # = 86,016
    return rm


def make_row_id_map(routing_map):
    """routing_map [T,E] bool -> row_id_map [T,E] int64（non-trans，-1 = 未选中）。"""
    mask = routing_map.to(torch.int64)
    within = torch.cumsum(mask, dim=0) - 1
    counts = mask.sum(dim=0)
    offset = torch.cumsum(counts, dim=0) - counts
    return torch.where(mask > 0, within + offset.unsqueeze(0),
                       torch.full_like(within, -1))


class Model(nn.Module):
    """routing_map [T,E] -> (row_id_map [E,T] int64, row_id_map_non_trans [T,E] int64)。"""

    def __init__(self):
        super().__init__()

    def forward(self, routing_map: torch.Tensor):
        row = make_row_id_map(routing_map)
        return row.t().contiguous(), row.contiguous()


def get_init_inputs():
    return []


def get_inputs():
    return [make_routing_map()]
