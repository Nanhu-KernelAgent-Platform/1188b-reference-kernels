"""case06 — CP 序列块重排（torch.index_select，被 torch.compile 融合）:
          triton_poi_fused_index_select_0

CSV   : 021-1188B-real-ops.csv / sheet 测试对象2 / 第 6 行
kernel: triton_poi_fused_index_select_0（没有独立源文件；由 torch.compile 从 torch.index_select 融合而来）
上层   : AttnFuncWithCPAndQKVOA2A（transformer_engine/pytorch/attention.py:4134；forward 在 :4263、
         backward 在 :4595 调 flash_attn_a2a_communicate，后者在 :1722/:1749 调
         reorder_seq_chunks_for_a2a，index_select 就在 :1685（按 seq 维）/:1691（按第 0 维））
索引构造: get_seq_chunk_ids_for_reordering（attention.py:1660）

真实形状（per-rank，CP=1 时）:
  k_ag [2, 2048, 1, 128, 192]  bf16   # 2*cp_size 个 seq chunk × (s//2) × b × heads × qk_dim
  v_ag [2, 2048, 1, 128, 128]  bf16
  index        [2]             int32  # cp_size=1 -> [0, 1]
  ----
  k_ag' [2, 2048, 1, 128, 192]  bf16  # = index_select(k_ag, 0, index)
  v_ag' [2, 2048, 1, 128, 128]  bf16

说明:
  * CP 把序列切成 2*cp_size 块、每卡拿不连续的两块；计算前需要按 chunk_ids 重排，
    这就是这两次 index_select 的用途（cp_size=1 时是恒等重排，但张量仍被整块拷贝）。
  * 由 CSV 数据自洽性可知规模正确: 平均 3.64 ms/次，读写 2 份 201MB 张量 ≈ 402MB
    ⇒ ~110 GB/s，与同表 cast/permute 的带宽量级一致。
"""

import torch
import torch.nn as nn

# ---- real per-rank dims (pretrain_1188B_formal_training.sh) ----
SEQ = 4096
BATCH = 1
NHEADS = 128
DQK = 192
DV = 128
CP = 1
CHUNKS = 2 * CP              # 2*cp_size
CHUNK_LEN = SEQ // 2         # s//2

CASE = {
    "id": 6,
    "name": "cp_seq_chunk_index_select",
    "kernel": "triton_poi_fused_index_select_0",
    "upstream": "AttnFuncWithCPAndQKVOA2A",
    "csv_row": 6,
    "source": "transformer_engine/pytorch/attention.py:1685,1691 (index_select inside "
              "reorder_seq_chunks_for_a2a, called at :1722/:1749 from flash_attn_a2a_communicate, "
              "invoked at :4263 fwd / :4595 bwd of class AttnFuncWithCPAndQKVOA2A); "
              "chunk ids: attention.py:1660",
    "dtype": "bf16 / index int32",
    "provenance": "形状由源码直接给出（高）；规模与 CSV 的 3.64ms/次带宽自洽（高）",
    "backward": False,
}


class Model(nn.Module):
    """返回 (index_select(k_ag, 0, index), index_select(v_ag, 0, index))."""

    def __init__(self):
        super().__init__()

    def forward(self, k_ag: torch.Tensor, v_ag: torch.Tensor, index: torch.Tensor):
        return (torch.index_select(k_ag, 0, index),
                torch.index_select(v_ag, 0, index))


def get_init_inputs():
    return []


def get_inputs():
    k_ag = torch.randn(CHUNKS, CHUNK_LEN, BATCH, NHEADS, DQK).to(torch.bfloat16)
    v_ag = torch.randn(CHUNKS, CHUNK_LEN, BATCH, NHEADS, DV).to(torch.bfloat16)
    # get_seq_chunk_ids_for_reordering(cp_size=1, to_contiguous=True) -> [0, 1]
    index = torch.arange(CHUNKS, dtype=torch.int32)
    return [k_ag, v_ag, index]
