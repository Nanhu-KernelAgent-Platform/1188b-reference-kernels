"""case08 — MoE grouped GEMM 反向 dgrad, fc2: dX 每组 [*, 2560]

CSV   : 021-1188B-real-ops.csv / sheet 测试对象2 / 第 8 行
kernel: musa_asm_bf16bf16bf16bf16ssgemm_nn_tce_512_256x256B128_epilogue_persis_stage2
上层   : _GroupedLinearBackward
来源   : transformer_engine/pytorch/module/grouped_linear.py:57,372; transformer_engine/musa/pytorch/csrc/extensions/gemm.cpp:296

真实形状（per-rank，TP=1 / CP=1 / micro-batch=1，见 model_config.json）:
  dy  [4096, N]      bf16
  w   [G, N, K]      bf16
  dX  [4096, K]      bf16

布局与语义（from repo）:
  TE backward 的 dgrad 用 layout "NN"（grouped_linear.py:372-382）,
  vendor asm 名里的 "nn" ⇒ dx[M,K] = dy[M,N] @ W[N,K]，K=out_features。
  x 第 i 段（每 group ~102 行（4096/40））与第 i 个 expert 的权重相乘。

置信度: shape 由模型配置推导（高）；asm 名 nn ⇒ dgrad（高）；fc2 归属（中）
"""

import torch
import torch.nn as nn

# ---- real per-rank dims (pretrain_1188B_formal_training.sh) ----
NUM_GROUPS = 40
TOKENS = 4096
N = 8192                 # fc2 out_features
K = 2560                 # fc2 in_features

CASE = {
    "id": 8,
    "name": "moe_grouped_gemm_fc2_dgrad",
    "kernel": "musa_asm_bf16bf16bf16bf16ssgemm_nn_tce_512_256x256B128_epilogue_persis_stage2",
    "upstream": "_GroupedLinearBackward",
    "csv_row": 8,
    "source": "transformer_engine/pytorch/module/grouped_linear.py:57,372; transformer_engine/musa/pytorch/csrc/extensions/gemm.cpp:296",
    "dtype": "bf16 in / fp32 accumulate / bf16 out",
    "provenance": "shape 由模型配置推导（高）；asm 名 nn ⇒ dgrad（高）；fc2 归属（中）",
    "backward": False,
}


def group_sizes(total, groups):
    """Deterministic uniform m_splits, sum(sizes) == total（真实训练里 distribution 不均匀）."""
    base, rem = divmod(total, groups)
    return [base + (1 if i < rem else 0) for i in range(groups)]


class Model(nn.Module):
    """dX_i = dY[seg_i] @ W_i   — per-expert input grad (NN 布局)"""

    def __init__(self, num_groups=NUM_GROUPS, tokens=TOKENS):
        super().__init__()
        self.num_groups = num_groups
        self.sizes = group_sizes(tokens, num_groups)

    def forward(self, dy: torch.Tensor, w: torch.Tensor):
        # dy [T, N], w [G, N, K]  ->  dx [T, K]
        outs = []
        off = 0
        for i, m in enumerate(self.sizes):
            outs.append(torch.mm(dy[off:off + m], w[i]))
            off += m
        return torch.cat(outs, dim=0)


def get_init_inputs():
    return [NUM_GROUPS, TOKENS]


def get_inputs():
    return [torch.randn(TOKENS, N).to(torch.bfloat16),
            torch.randn(NUM_GROUPS, N, K).to(torch.bfloat16)]
