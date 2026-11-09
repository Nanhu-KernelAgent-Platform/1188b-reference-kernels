"""case10 — grouped GEMM forward, 单 group 覆盖全部 4096 行（N=5120）

CSV   : 021-1188B-real-ops.csv / sheet 测试对象2 / 第 10 行
kernel: musa_asm_bf16bf16bf16bf16gemm_nt_tce_768_256x320B128_squad_level_epilogue
上层   : _GroupedLinear / CheckpointFunction
来源   : transformer_engine/pytorch/module/grouped_linear.py:57,168; transformer_engine/musa/pytorch/csrc/extensions/gemm.cpp:296

真实形状（per-rank，TP=1 / CP=1 / micro-batch=1，见 model_config.json）:
  x   [4096, K]      bf16
  w   [G, N, K]      bf16        # G=40(local experts) 或 1
  out [4096, N]      bf16

布局与语义（from repo）:
  TE forward 调用 general_grouped_gemm(..., m_splits=...)（grouped_linear.py:168-179），
  权重布局 [out, in]；vendor asm 名里的 "nt" = D = op(B) @ op(A) = x[M,K] @ W[N,K]^T，即前向。
  x 第 i 段（全部 4096 行）与第 i 个 expert 的权重相乘。

置信度: shape 高（由 N/K 推断同 fc1）；**归属低**：320-tile 这 4 条 （10/13/14/19）在仓库里查不到名字，候选解释见 README §4（shared-expert / 单组 GEMM）
"""

import torch
import torch.nn as nn

# ---- real per-rank dims (pretrain_1188B_formal_training.sh) ----
NUM_GROUPS = 1
TOKENS = 4096
N = 5120
K = 8192

CASE = {
    "id": 10,
    "name": "grouped_gemm_single_group_fc1_fwd",
    "kernel": "musa_asm_bf16bf16bf16bf16gemm_nt_tce_768_256x320B128_squad_level_epilogue",
    "upstream": "_GroupedLinear / CheckpointFunction",
    "csv_row": 10,
    "source": "transformer_engine/pytorch/module/grouped_linear.py:57,168; transformer_engine/musa/pytorch/csrc/extensions/gemm.cpp:296",
    "dtype": "bf16 in / fp32 accumulate / bf16 out",
    "provenance": "shape 高（由 N/K 推断同 fc1）；**归属低**：320-tile 这 4 条 （10/13/14/19）在仓库里查不到名字，候选解释见 README §4（shared-expert / 单组 GEMM）",
    "backward": False,
}


def group_sizes(total, groups):
    """Deterministic uniform m_splits, sum(sizes) == total（真实训练里 distribution 不均匀）."""
    base, rem = divmod(total, groups)
    return [base + (1 if i < rem else 0) for i in range(groups)]


class Model(nn.Module):
    """out = x @ W[0].T   — 1 个 group（4096 行一次算完）"""

    def __init__(self, num_groups=NUM_GROUPS, tokens=TOKENS):
        super().__init__()
        self.num_groups = num_groups
        self.sizes = group_sizes(tokens, num_groups)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        # x [T, K], w [G, N, K]  ->  [T, N]
        outs = []
        off = 0
        for i, m in enumerate(self.sizes):
            outs.append(torch.mm(x[off:off + m], w[i].t()))
            off += m
        return torch.cat(outs, dim=0)


def get_init_inputs():
    return [NUM_GROUPS, TOKENS]


def get_inputs():
    return [torch.randn(TOKENS, K).to(torch.bfloat16),
            torch.randn(NUM_GROUPS, N, K).to(torch.bfloat16)]
