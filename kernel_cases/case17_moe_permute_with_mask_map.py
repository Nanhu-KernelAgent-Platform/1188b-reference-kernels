"""case17 — MoE permute with mask map (+ permuted probs):  permute_with_mask_map<bf16,float,long,true>

CSV   : 021-1188B-real-ops.csv / sheet 测试对象2 / 第 17 行
kernel: permute_with_mask_map<__mt_bfloat16, float, long, true>   （最后一个 bool = with_permuted_probs）
上层   : _moe_permute_mask_map（transformer_engine/pytorch/permutation.py:255，dispatch 掉 tes.moe_permute_mask :312）
Kernel : transformer_engine/musa/common/permutation/permutation_mask.mu:63-125

真实形状（per-rank）:
  x          [4096, 8192]   bf16    # num_tokens=4096, hidden=8192
  row_id_map [4096, 320]    int64   # non-trans 布局（token-major），非选位置 = -1，值 = 目标行号
  probs      [4096, 320]    fp32    # 稀疏：每行只有 topk=8 个非零
  ----
  out            [32768, 8192]  bf16    # 32768 = 4096*topk(8)
  permuted_probs [32768]        fp32

语义: 每个 token 行被复制到它选中的每个 expert 的目标行（scatter），同时把 probs[t][e] 写到同一目标行。
      out[row_id_map[t][e]] = x[t], permuted_probs[row_id_map[t][e]] = probs[t][e]  (row_id_map[t][e] != -1)
"""

import torch
import torch.nn as nn

# ---- real per-rank dims (pretrain_1188B_formal_training.sh) ----
TOKENS = 4096
HID = 8192
NUM_EXPERTS = 320
TOPK = 8
NUM_OUT_TOKENS = TOKENS * TOPK     # 32768

CASE = {
    "id": 17,
    "name": "moe_permute_with_mask_map",
    "kernel": "permute_with_mask_map<__mt_bfloat16, float, long, true>",
    "upstream": "_moe_permute_mask_map",
    "csv_row": 17,
    "source": "transformer_engine/pytorch/permutation.py:255; "
              "transformer_engine/musa/common/permutation/permutation_mask.mu:63",
    "dtype": "bf16 act / fp32 probs / int64 map",
    "provenance": "形状由源码+配置直接给出（高）：num_out_tokens = num_tokens*topk（"
                  "Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py:550）",
    "backward": False,
}


def make_row_id_map(routing_map):
    """routing_map [T,E] bool -> row_id_map [T,E] int64（non-trans，-1 = 未选中）。

    = 每个 expert 段内的序号 + 该 expert 段的全局起始行（各 expert token 数的前缀和）。
    """
    mask = routing_map.to(torch.int64)
    within = torch.cumsum(mask, dim=0) - 1                 # [T,E] 段内序号（含自己）
    counts = mask.sum(dim=0)                               # [E] 每个 expert 的 token 数
    offset = torch.cumsum(counts, dim=0) - counts          # [E] 每段全局起始行
    return torch.where(mask > 0, within + offset.unsqueeze(0),
                       torch.full_like(within, -1))



class Model(nn.Module):
    """out[dst] = x[t]（dst=row_id_map[t,e]>=0）；permuted_probs[dst] = probs[t,e]."""

    def __init__(self, num_experts=NUM_EXPERTS, num_out_tokens=NUM_OUT_TOKENS):
        super().__init__()
        self.num_experts = num_experts
        self.num_out = num_out_tokens

    def forward(self, x: torch.Tensor, row_id_map: torch.Tensor, probs: torch.Tensor):
        n, e = row_id_map.shape
        dst = row_id_map.reshape(-1)                         # token-major 展平
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
    return [NUM_EXPERTS, NUM_OUT_TOKENS]


def get_inputs():
    rm = torch.zeros(TOKENS, NUM_EXPERTS, dtype=torch.bool)
    idx = torch.argsort(torch.rand(TOKENS, NUM_EXPERTS), dim=-1)[:, :TOPK]
    rm.scatter_(1, idx, True)
    row_id_map = make_row_id_map(rm)
    x = torch.randn(TOKENS, HID).to(torch.bfloat16)
    probs = torch.zeros(TOKENS, NUM_EXPERTS, dtype=torch.float32)
    probs.scatter_(1, idx, torch.rand(TOKENS, TOPK, dtype=torch.float32))
    return [x, row_id_map, probs]
