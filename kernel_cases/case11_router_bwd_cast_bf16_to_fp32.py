"""case11 — bf16 → fp32 cast（aten::_copy_from）在 RouterGatingLinearFunctionBackward 里

CSV   : 021-1188B-real-ops.csv / sheet 测试对象2 / 第 11 行
kernel: musa::dnn::unary::kernel_unary<__mt_bfloat16, musa::dnn::unary::CastOp, float, float, __mt_bfloat16, 128>
上层   : RouterGatingLinearFunctionBackward（Megatron-LM/megatron/core/transformer/moe/moe_utils.py:1598，
         forward/backward 调 te_general_gemm 在第 1631、1664 行；Cast kernel 来自该处的 .to()）
真实形状（推定）: in [4096, 8192] bf16  ->  out [4096, 8192] fp32

为什么是这个形状（依据 CSV 自身的数据反推）:
  该 kernel 平均 2.07 ms/次；把 [4096, 8192] bf16 读 + fp32 写 = 64 MB + 128 MB = 192 MB，
  对应 ~93 GB/s，与 CSV 里其它纯访存 kernel 的带宽同量级（aten::add 0.39ms→~490GB/s，
  permute 3.11ms→~185GB/s）。若换成 [4096, 320] 或 [320, 8192]，带宽会低到 3~8 GB/s，不合理。

语义: 纯数据类型转换（bf16 -> fp32，fp32 是 router 的计算精度，`--moe-router-dtype fp32`）。
"""

import torch
import torch.nn as nn

# ---- real per-rank dims (pretrain_1188B_formal_training.sh) ----
TOKENS = 4096            # seq 4096 * micro-batch 1
HID = 8192               # hidden_size

CASE = {
    "id": 11,
    "name": "router_bwd_cast_bf16_to_fp32",
    "kernel": "musa::dnn::unary::kernel_unary<__mt_bfloat16, musa::dnn::unary::CastOp, float, float, __mt_bfloat16, 128>",
    "upstream": "RouterGatingLinearFunctionBackward / aten::_copy_from",
    "csv_row": 11,
    "source": "Megatron-LM/megatron/core/transformer/moe/moe_utils.py:1598 (RouterGatingLinearFunction)",
    "dtype": "bf16 -> fp32",
    "provenance": "形状由 CSV 的‘平均耗时 2.07ms + 访存带宽自洽’反推（中）；语义=cast（高）",
    "backward": False,
}


class Model(nn.Module):
    """.to(torch.float32) —— router 反向里的 bf16→fp32 cast。"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)


def get_init_inputs():
    return []


def get_inputs():
    return [torch.randn(TOKENS, HID).to(torch.bfloat16)]
