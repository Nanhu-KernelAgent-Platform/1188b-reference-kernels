"""case12 — MoE row_id_map 构建（第 2 遍）: _row_id_map_pass_2_kernel

CSV   : 021-1188B-real-ops.csv / sheet 测试对象2 / 第 12 行
kernel: _row_id_map_pass_2_kernel
上层   : _moe_permute_mask_map（transformer_engine/pytorch/permutation.py:255，其中 :287 调
         triton_permutation.make_row_id_map）
来源   : transformer_engine/pytorch/triton/permutation.py:49-85（pass 2）、:17-45（pass 1）、
         :88-119（make_row_id_map 驱动）

真实形状（per-rank）:
  routing_map (bool)        [4096, 320]      # num_tokens=4096（seq 4096 x mbs 1）, num_experts=320
  row_id_map  (int64)       [320, 4096]      # expert-major（trans），值 = 该 token 在本 expert 段内的行号，-1 = 未被选中
  row_id_map_non_trans      [4096, 320]      # token-major（同一批值转置写出）
  workspace   (int64)       [320, 16]        # 每 (expert, 1024-token block) 的计数，pass1 的产出

每个 token 选中 topk=8 个 expert -> 有效 (token,expert) 对数 = 32768 = permute 的输出行数
（4096*8；所以 row_id_map 的取值范围是 [0, 32768)）。

语义（两个 pass 的合并语义，等价于“把 1 的计数沿 expert 做前缀和再减 1，0 -> -1”）:
  pass1: 每个 (expert, token-block) 内做 block 内 inclusive cumsum，并把 block 内计数存到 workspace
  pass2: row_id[e,t] = local_cumsum[e,t] + sum(workspace 中位于本 block 之前的计数) - 1, mask==0 处写 -1
  注意 workspace 是 [num_experts, num_blocks] 行主序展平，chunk_idx = e*nblocks + b，
  所以前缀和里既包含本 expert 之前的 block，也包含**前面所有 expert 的整段**。
  ⇒ 参考实现必须写成「段内 cumsum - 1 + 该 expert 段全局起始行」，不能只做一次全局 cumsum。
"""

import torch
import torch.nn as nn

# ---- real per-rank dims (pretrain_1188B_formal_training.sh) ----
TOKENS = 4096
NUM_EXPERTS = 320
TOPK = 8

CASE = {
    "id": 12,
    "name": "moe_row_id_map_pass2",
    "kernel": "_row_id_map_pass_2_kernel",
    "upstream": "_moe_permute_mask_map",
    "csv_row": 12,
    "source": "transformer_engine/pytorch/triton/permutation.py:49 (_row_id_map_pass_2_kernel), "
              "driven by make_row_id_map:88, called from transformer_engine/pytorch/permutation.py:287",
    "dtype": "routing_map bool -> row_id_map int64",
    "provenance": "形状由 num_tokens/num_experts/topk 直接决定（高）；语义=源码直读（高）",
    "backward": False,
}


def make_row_id_map(routing_map):
    """routing_map [T,E] bool -> row_id_map [T,E] int64（non-trans，-1 = 未选中）。"""
    mask = routing_map.to(torch.int64)
    within = torch.cumsum(mask, dim=0) - 1                 # [T,E] 段内序号
    counts = mask.sum(dim=0)                               # [E] 每个 expert 的 token 数
    offset = torch.cumsum(counts, dim=0) - counts          # [E] 每段全局起始行
    return torch.where(mask > 0, within + offset.unsqueeze(0),
                       torch.full_like(within, -1))


class Model(nn.Module):
    """routing_map [T,E] (0/1) -> (row_id_map [E,T] int64, row_id_map_non_trans [T,E] int64)."""

    def __init__(self):
        super().__init__()

    def forward(self, routing_map: torch.Tensor):
        row = make_row_id_map(routing_map)
        return row.t().contiguous(), row.contiguous()


def get_init_inputs():
    return []


def get_inputs():
    rm = torch.zeros(TOKENS, NUM_EXPERTS, dtype=torch.bool)
    idx = torch.argsort(torch.rand(TOKENS, NUM_EXPERTS), dim=-1)[:, :TOPK]   # 每 token 选 topk=8
    rm.scatter_(1, idx, True)
    return [rm]
