"""case09 — MoE grouped GEMM 反向 wgrad, fc1: dW 每组 [5120, 8192]

CSV   : 021-1188B-real-ops.csv / sheet 测试对象2 / 第 9 行
kernel: musa_asm_bf16bf16bf16bf16ssgemm_tn_tce_768_384x256B128_epilogue_mldz_persis_stage2
上层   : _GroupedLinearBackward
来源   : transformer_engine/pytorch/module/grouped_linear.py:57,411; transformer_engine/musa/pytorch/csrc/extensions/gemm.cpp:296

真实形状（per-rank，TP=1 / CP=1 / micro-batch=1，见 model_config.json）:
  dy  [4096, N]      bf16
  x   [4096, K]      bf16
  dW  [G, N, K]      bf16

布局与语义（from repo）:
  TE backward 的 wgrad 用 layout "NT"（grouped_linear.py:411-424）,
  vendor asm 名里的 "tn" ⇒ dW[N,K] = dy[M,N]^T @ x[M,K]，按 expert 分组累加（每个 group 单独出 dW）。
  x 第 i 段（每 group ~102 行（4096/40））与第 i 个 expert 的权重相乘。

置信度: shape 由模型配置推导（高）；asm 名 tn ⇒ wgrad（高）；fc1 归属（中）
"""

import torch
import torch.nn as nn

# ---- real per-rank dims (pretrain_1188B_formal_training.sh) ----
NUM_GROUPS = 40
TOKENS = 4096
N = 5120                 # fc1 out_features
K = 8192                 # hidden_size

CASE = {
    "id": 9,
    "name": "moe_grouped_gemm_fc1_wgrad",
    "kernel": "musa_asm_bf16bf16bf16bf16ssgemm_tn_tce_768_384x256B128_epilogue_mldz_persis_stage2",
    "upstream": "_GroupedLinearBackward",
    "csv_row": 9,
    "source": "transformer_engine/pytorch/module/grouped_linear.py:57,411; transformer_engine/musa/pytorch/csrc/extensions/gemm.cpp:296",
    "dtype": "bf16 in / fp32 accumulate / bf16 out",
    "provenance": "shape 由模型配置推导（高）；asm 名 tn ⇒ wgrad（高）；fc1 归属（中）",
    "backward": False,
}


def group_sizes(total, groups):
    """Deterministic uniform m_splits, sum(sizes) == total（真实训练里 distribution 不均匀）."""
    base, rem = divmod(total, groups)
    return [base + (1 if i < rem else 0) for i in range(groups)]


class Model(nn.Module):
    """dW_i = dY[seg_i].T @ X[seg_i]   — per-expert weight grad (TN 布局)"""

    def __init__(self, num_groups=NUM_GROUPS, tokens=TOKENS):
        super().__init__()
        self.num_groups = num_groups
        self.sizes = group_sizes(tokens, num_groups)

    def forward(self, dy: torch.Tensor, x: torch.Tensor):
        # dy [T, N], x [T, K]  ->  dw [G, N, K]
        outs = []
        off = 0
        for m in self.sizes:
            outs.append(torch.mm(dy[off:off + m].t(), x[off:off + m]))
            off += m
        return torch.stack(outs, dim=0)


def get_init_inputs():
    return [NUM_GROUPS, TOKENS]


def get_inputs():
    return [torch.randn(TOKENS, N).to(torch.bfloat16),
            torch.randn(TOKENS, K).to(torch.bfloat16)]
