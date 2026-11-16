"""case17（真实形状版）— MoE permute with mask map (+permuted probs): permute_with_mask_map<bf16,float,long,true>

形状来源（唯一权威）: 021-1188B-real-ops-11-11-record.xlsx
  trace: x1w_1188B_128k_ac-full_rank-8.json（seq 131072 / CP 8 / 每卡 16384 tokens）
  记录: x(16384,8192) + map(16384,320) + probs(16384,320) -> y(R,8192) + pp(R)，R=80,820-99,179/层
  profiler 真实平均: 3.11 ms
本版构造: 与 case12 同一确定性 routing_map，R=86,016；probs 稀疏结构与真实一致（每行仅选中处非零）。
注意: 旧 kernel_cases 版用 TOKENS=4096/TOPK=8（R=32768 满 dispatch），形状作废。
"""

import torch
import torch.nn as nn

# ---- real per-rank dims (xlsx record, 128k seq / CP 8) ----
TOKENS = 16384
HID = 8192
NUM_EXPERTS = 320

CASE = {
    "id": 17,
    "name": "moe_permute_with_mask_map",
    "kernel": "permute_with_mask_map<__mt_bfloat16, float, long, true>",
    "csv_row": 17,
    "profiler_avg_ms": 3.11,
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
    return rm, sel


def make_row_id_map(routing_map):
    mask = routing_map.to(torch.int64)
    within = torch.cumsum(mask, dim=0) - 1
    counts = mask.sum(dim=0)
    offset = torch.cumsum(counts, dim=0) - counts
    return torch.where(mask > 0, within + offset.unsqueeze(0), torch.full_like(within, -1))


class Model(nn.Module):
    """out[dst]=x[t]（dst=row_id_map[t,e]>=0）；permuted_probs[dst]=probs[t,e]。"""

    def __init__(self, num_experts=NUM_EXPERTS, num_out_tokens=86016):
        super().__init__()
        self.num_experts = num_experts
        self.num_out = num_out_tokens

    def forward(self, x: torch.Tensor, row_id_map: torch.Tensor, probs: torch.Tensor):
        n, e = row_id_map.shape
        dst = row_id_map.reshape(-1)
        keep = dst >= 0
        tok = torch.arange(n, device=x.device).repeat_interleave(e)
        src_rows = tok[keep]
        dst_rows = dst[keep]

        out = x.new_zeros((self.num_out, x.shape[1]))
        out.index_copy_(0, dst_rows, x.index_select(0, src_rows))

        permuted_probs = probs.new_zeros(self.num_out)
        permuted_probs.index_copy_(0, dst_rows, probs.reshape(-1)[keep])
        return out, permuted_probs


def get_init_inputs():
    return [NUM_EXPERTS, 86016]


def get_inputs():
    rm, sel = make_routing_map()
    row_id_map = make_row_id_map(rm)
    r_rows = int(rm.sum())
    assert r_rows == 86016
    torch.manual_seed(20261114)
    x = torch.randn(TOKENS, HID).to(torch.bfloat16)
    probs = torch.zeros(TOKENS, NUM_EXPERTS, dtype=torch.float32)
    vals = torch.rand(TOKENS, 5, dtype=torch.float32)
    probs.scatter_(1, sel[:, :5], vals)
    extra = (torch.arange(TOKENS) % 4 == 1)
    probs[extra] = probs[extra].scatter(1, sel[extra, 5:6], torch.rand(int(extra.sum()), 1))
    return [x, row_id_map, probs]
