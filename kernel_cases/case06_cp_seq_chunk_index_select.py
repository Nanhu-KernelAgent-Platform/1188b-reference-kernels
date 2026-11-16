"""case06（真实形状版）— CP 序列块重排: triton_poi_fused_index_select_0

形状来源（唯一权威）: 021-1188B-real-ops-11-11-record.xlsx
  trace: x1w_1188B_128k_ac-full_rank-8.json（seq 131072 / CP 8 / TP 8 / 每卡 16384 tokens）
  记录: k_ag(2,65536,16,192) + v_ag(2,65536,16,128) + idx(2) -> 同形状 2 个
  profiler 真实平均: 3.64 ms（543.26ms 等列见 csv 第 6 行）
  自洽校核: 读+写 2 x (805+537)MB = 2.68GB；3.64ms -> 737 GB/s，带宽量级合理。
注意: 旧 kernel_cases 版误用 seq 4096/CP 1 的小形状（[2,2048,1,128,192]），作废。
"""

import torch
import torch.nn as nn

# ---- real per-rank dims (xlsx record, 128k seq / CP 8 / TP 8) ----
CHUNKS = 2
TOK = 65536      # 131072/2
HEADS = 16       # 128 heads / TP 8
DQK = 192
DV = 128

CASE = {
    "id": 6,
    "name": "cp_seq_chunk_index_select",
    "kernel": "triton_poi_fused_index_select_0",
    "csv_row": 6,
    "profiler_avg_ms": 3.64,
    "shape_provenance": "021-1188B-real-ops-11-11-record.xlsx (x1w_1188B_128k_ac-full_rank-8.json)",
    "backward": False,
}


class Model(nn.Module):
    """(index_select(k_ag,0,idx), index_select(v_ag,0,idx)) —— CP= 真实形状下逐元素重排拷贝。"""

    def __init__(self):
        super().__init__()

    def forward(self, k_ag: torch.Tensor, v_ag: torch.Tensor, index: torch.Tensor):
        return (torch.index_select(k_ag, 0, index),
                torch.index_select(v_ag, 0, index))


def get_init_inputs():
    return []


def get_inputs():
    torch.manual_seed(20261111)
    k_ag = torch.randn(CHUNKS, TOK, HEADS, DQK).to(torch.bfloat16)
    v_ag = torch.randn(CHUNKS, TOK, HEADS, DV).to(torch.bfloat16)
    index = torch.arange(CHUNKS, dtype=torch.int32)
    return [k_ag, v_ag, index]
