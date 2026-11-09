"""case16 — bf16 逐元素 add:  musa::dnn::binary::kernel_binary<__mt_bfloat16, AddOp, float, bf16, bf16, 128>

CSV   : 021-1188B-real-ops.csv / sheet 测试对象2 / 第 16 行
kernel: musa::dnn::binary::kernel_binary<__mt_bfloat16, musa::dnn::binary::AddOp, float, __mt_bfloat16, __mt_bfloat16, 128>
上层   : aten::add（CSV 位置列写的是「aten算子」，即普通 aten 逐元素加，无对应仓库源码）
真实形状（推定）: a [4096, 8192] bf16 + b [4096, 8192] bf16 -> out [4096, 8192] bf16

为什么是这个形状: 平均 0.39 ms/次。两份 [4096, 8192] bf16 读 + 一份写 = 192 MB，
   对应 ~490 GB/s —— 与同表其它纯访存 kernel（cast 2.07ms→~93GB/s、permute 3.11ms→~185GB/s）
   的量级关系自洽；若换成 [4096, 320] 之类小张量，带宽会低两个数量级。

语义: out = a + b（bf16 计算，fp32 累加由 kernel 内部的 float 模板参数保证）。
"""

import torch
import torch.nn as nn

# ---- real per-rank dims (pretrain_1188B_formal_training.sh) ----
TOKENS = 4096
HID = 8192

CASE = {
    "id": 16,
    "name": "aten_add_bf16",
    "kernel": "musa::dnn::binary::kernel_binary<__mt_bfloat16, musa::dnn::binary::AddOp, float, __mt_bfloat16, __mt_bfloat16, 128>",
    "upstream": "aten::add",
    "csv_row": 16,
    "source": "aten 算子（无仓库源码；CSV 位置列=‘aten算子’）",
    "dtype": "bf16",
    "provenance": "形状由 CSV 平均耗时+带宽自洽反推（中，见文件头）；语义=逐元素 add（高）",
    "backward": False,
}


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


def get_init_inputs():
    return []


def get_inputs():
    return [torch.randn(TOKENS, HID).to(torch.bfloat16),
            torch.randn(TOKENS, HID).to(torch.bfloat16)]
