"""case03 — MoE grouped GEMM forward, fc1(gate_up), 40 组 (每个 local expert 一组)

CSV   : 021-1188B-real-ops.csv / sheet 测试对象2 / 第 3 行
kernel: musa_asm_bf16bf16bf16bf16ssgemm_nt_tce_768_384x256B128_epilogue_persis_stage2
上层   : _GroupedLinear / CheckpointFunction
来源   : transformer_engine/pytorch/module/grouped_linear.py:57,168; transformer_engine/musa/pytorch/csrc/extensions/gemm.cpp:296 (te_general_grouped_gemm)

真实形状（per-rank，TP=1 / CP=1 / micro-batch=1，见 model_config.json）:
  x   [4096, K]      bf16
  w   [G, N, K]      bf16        # G=40(local experts) 或 1
  out [4096, N]      bf16

布局与语义（from repo）:
  TE forward 调用 general_grouped_gemm(..., m_splits=...)（grouped_linear.py:168-179），
  权重布局 [out, in]；vendor asm 名里的 "nt" = D = op(B) @ op(A) = x[M,K] @ W[N,K]^T，即前向。
  x 第 i 段（每 group ~102 行（4096/40））与第 i 个 expert 的权重相乘。

置信度: shape 由模型配置推导（高）；asm 名 nt ⇒ 前向（高）；fc1 归属（中：由 N/K 与 tile 家族配对推断，见 README §4）
"""

import torch
import torch.nn as nn

# ---- real per-rank dims (pretrain_1188B_formal_training.sh) ----
NUM_GROUPS = 40          # local experts = num_experts 320 / EP 8
TOKENS = 4096            # seq 4096 * micro-batch 1 (TP=1)
N = 5120                 # 2 * moe_ffn_hidden_size(2560) = gate+up
K = 8192                 # hidden_size

CASE = {
    "id": 3,
    "name": "moe_grouped_gemm_fc1_fwd",
    "kernel": "musa_asm_bf16bf16bf16bf16ssgemm_nt_tce_768_384x256B128_epilogue_persis_stage2",
    "upstream": "_GroupedLinear / CheckpointFunction",
    "csv_row": 3,
    "source": "transformer_engine/pytorch/module/grouped_linear.py:57,168; transformer_engine/musa/pytorch/csrc/extensions/gemm.cpp:296 (te_general_grouped_gemm)",
    "dtype": "bf16 in / fp32 accumulate / bf16 out",
    "provenance": "shape 由模型配置推导（高）；asm 名 nt ⇒ 前向（高）；fc1 归属（中：由 N/K 与 tile 家族配对推断，见 README §4）",
    "backward": False,
}


def group_sizes(total, groups):
    """Deterministic uniform m_splits, sum(sizes) == total（真实训练里 distribution 不均匀）."""
    base, rem = divmod(total, groups)
    return [base + (1 if i < rem else 0) for i in range(groups)]


class Model(nn.Module):
    """out = cat_i( x[seg_i] @ W[i].T )   — grouped GEMM forward, NT 布局"""

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
