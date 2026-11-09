"""case02 — MLA/FlashAttention forward（causal, softmax fp32, bf16）:
          musa_asm_bf16tce_flash_atten_512_256x64_192_128_causal_persistence

CSV   : 021-1188B-real-ops.csv / sheet 测试对象2 / 第 2 行
kernel: musa_asm_bf16tce_flash_atten_512_256x64_192_128_causal_persistence
上层   : AttnFuncWithCPAndQKVOA2A（transformer_engine/pytorch/attention.py:4134；MUSA 启动点
         transformer_engine/musa/pytorch/csrc/extensions/attention.mu:573,617）
形状来源: Megatron-LM/megatron/core/transformer/multi_latent_attention.py:864-901
         （kv 上投影 + MLA rope 之后得到完整 key/value，再交给 TE 的注意力核）

真实形状（per-rank，TP=1 / CP=1 / micro-batch=1 / sbhd）:
  q [4096, 1, 128, 192]  bf16      # 192 = qk_head_dim(128) + qk_pos_emb_head_dim(64)
  k [4096, 1, 128, 192]  bf16
  v [4096, 1, 128, 128]  bf16      # v_head_dim = 128
  ----
  o [4096, 1, 128, 128]  bf16

两套参考实现（默认 sdpa）: impl="sdpa" = torch SDPA（官方 flash-attn 路径，基准性能用它）；
impl="explicit" = 分块显式 softmax（用于交叉验证）。

语义: o = softmax(q @ k^T * scale + causal_mask) @ v,  scale = 1/sqrt(192),
      softmax 在 fp32（--attention-softmax-in-fp32），mscale = 1.0（yarn, rotary_scaling_factor=1）。
kernel 名里的 192/128 正是 qk 维与 v 维（另两个数字 512/256x64 是 asm 的 tile/dispatch 参数）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- real per-rank dims (pretrain_1188B_formal_training.sh) ----
SEQ = 4096
BATCH = 1
NHEADS = 128
DQK = 192          # 128 + 64
DV = 128

CASE = {
    "id": 2,
    "name": "mla_flash_attn_fwd_causal",
    "kernel": "musa_asm_bf16tce_flash_atten_512_256x64_192_128_causal_persistence",
    "upstream": "AttnFuncWithCPAndQKVOA2A",
    "csv_row": 2,
    "source": "transformer_engine/pytorch/attention.py:4134; "
              "transformer_engine/musa/pytorch/csrc/extensions/attention.mu:573,617; "
              "caller Megatron-LM/megatron/core/transformer/multi_latent_attention.py:892",
    "dtype": "bf16 in/out, fp32 softmax",
    "provenance": "形状由源码+配置直接给出（高）；kernel 名里的 qk_192/v_128 与 MLA 配置一致（高）",
    "backward": False,
}


def flash_attn_fwd(q, k, v, scale, causal=True, chunk=256):
    """q,k [S,B,H,Dqk]; v [S,B,H,Dv] -> o [S,B,H,Dv]（分块，fp32 softmax）."""
    s, b, h, _ = q.shape
    qp = q.permute(1, 2, 0, 3)                        # [B,H,S,D]
    kp = k.permute(1, 2, 0, 3)
    vp = v.permute(1, 2, 0, 3)
    kt = kp.transpose(-1, -2)
    o = torch.empty((b, h, s, v.shape[-1]), dtype=v.dtype, device=q.device)
    idx = torch.arange(s, device=q.device)
    for i0 in range(0, s, chunk):
        i1 = min(i0 + chunk, s)
        sc = torch.matmul(qp[:, :, i0:i1], kt).float() * scale     # [B,H,m,S]
        if causal:
            keep = idx[None, :] <= idx[i0:i1][:, None]
            sc = sc.masked_fill(~keep, float("-inf"))
        p = torch.softmax(sc, dim=-1)
        o[:, :, i0:i1] = torch.matmul(p.to(v.dtype), vp)
        del sc, p
    return o.permute(2, 0, 1, 3).contiguous()


class Model(nn.Module):
    """impl="sdpa"：torch 官方 flash-attention 路径（推荐基准/对照）；"explicit"：分块显式 softmax。"""

    def __init__(self, chunk=256, impl="sdpa"):
        super().__init__()
        self.chunk = chunk
        self.impl = impl

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scale = q.shape[-1] ** -0.5
        if self.impl == "sdpa":
            bshd = [x.permute(1, 2, 0, 3).contiguous() for x in (q, k, v)]
            o = F.scaled_dot_product_attention(*bshd, is_causal=True, scale=scale)
            return o.permute(2, 0, 1, 3).contiguous()
        return flash_attn_fwd(q, k, v, scale, causal=True, chunk=self.chunk)


def get_init_inputs():
    return [256, "sdpa"]


def get_inputs():
    q = torch.randn(SEQ, BATCH, NHEADS, DQK).to(torch.bfloat16)
    k = torch.randn(SEQ, BATCH, NHEADS, DQK).to(torch.bfloat16)
    v = torch.randn(SEQ, BATCH, NHEADS, DV).to(torch.bfloat16)
    return [q, k, v]
