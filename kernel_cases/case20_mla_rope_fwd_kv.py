"""case20 — MLA YaRN RoPE forward for K/V:  rotary_fwd_kv_kernel

CSV   : 021-1188B-real-ops.csv / sheet 测试对象2 / 第 20 行
kernel: rotary_fwd_kv_kernel
上层   : ApplyMLARotaryEmbKV（Megatron-LM/megatron/core/fusions/fused_mla_yarn_rope_apply.py:
         kernel 定义第 392 行，autograd.Function 第 595 行，wrapper 第 749 行）
调用点 : Megatron-LM/megatron/core/transformer/multi_latent_attention.py:892
         （self.config.apply_rope_fusion，kv 已经过 linear_kv_up_proj 上投影）

真实形状（per-rank，TP=1 / CP=1 / micro-batch=1 / sbhd 格式）:
  kv        [4096, 1, 128, 256]   bf16    # 256 = qk_head_dim(128) + v_head_dim(128)
  k_pos_emb [4096, 1, 1,   64]    bf16    # qk_pos_emb_head_dim = 64，按 head 广播
  cos / sin [4096, 1, 1,   64]    fp32    # YaRN 表，max_position_embeddings = seq 4096
  --------
  key       [4096, 1, 128, 192]   bf16    # 192 = qk_head_dim(128) + emb(64)
  value     [4096, 1, 128, 128]   bf16

语义（kernel 原文，fused_mla_yarn_rope_apply.py:425-489）:
  k = kv[..., :k_dim]        -> key[..., k_dim:]          （原样搬运）
  v = kv[..., k_dim:]        -> value[..., :]             （原样搬运）
  x  = k_pos_emb[..., :emb]  （每 head 广播）
  HALF-SPLIT（非 interleaved）旋转，even/odd 拆分:
      x1 = x[0::2], x2 = x[1::2]
      key[..., k_dim : k_dim+emb/2]     = x1*cos[:emb/2] - x2*sin[:emb/2]
      key[..., k_dim+emb/2 : k_dim+emb] = x2*cos[emb/2:] + x1*sin[emb/2:]
  注意: rope 结果放在 key 的 *尾部*，k 在前 —— 与 q 的布局（rope 在尾部）一致。
"""

import torch
import torch.nn as nn

# ---- real per-rank dims (pretrain_1188B_formal_training.sh) ----
SEQ = 4096
BATCH = 1
NHEADS = 128
K_DIM = 128        # qk_head_dim（k_nope 上投影后的 key 部分）
V_DIM = 128        # v_head_dim
EMB = 64           # qk_pos_emb_head_dim

CASE = {
    "id": 20,
    "name": "mla_rope_fwd_kv",
    "kernel": "rotary_fwd_kv_kernel",
    "upstream": "ApplyMLARotaryEmbKV",
    "csv_row": 20,
    "source": "Megatron-LM/megatron/core/fusions/fused_mla_yarn_rope_apply.py:392,595,749; "
              "call site Megatron-LM/megatron/core/transformer/multi_latent_attention.py:892",
    "dtype": "bf16 数据 / fp32 cos-sin 表",
    "provenance": "形状由源码 docstring + 调用点直接给出（高）；"
                  "seq=4096/max_position_embeddings=4096 来自训练脚本（高）",
    "backward": False,
}


class Model(nn.Module):
    """key = cat([kv[..., :K_DIM], rope(k_pos_emb)], -1), value = kv[..., K_DIM:]（rope 按 head 广播）."""

    def __init__(self, k_dim=K_DIM, v_dim=V_DIM, emb=EMB):
        super().__init__()
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.emb = emb

    def forward(self, kv: torch.Tensor, k_pos_emb: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor):
        k = kv[..., : self.k_dim]
        v = kv[..., self.k_dim: self.k_dim + self.v_dim]
        half = self.emb // 2
        x = k_pos_emb.float()                      # [S,B,1,emb]
        x1, x2 = x[..., 0::2], x[..., 1::2]         # even / odd
        c1, s1 = cos[..., :half], sin[..., :half]
        c2, s2 = cos[..., half:], sin[..., half:]
        left = x1 * c1 - x2 * s1
        right = x2 * c2 + x1 * s2
        rope = torch.cat([left, right], dim=-1)     # [S,B,1,emb]
        rope = rope.expand(-1, -1, k.shape[-2], -1)  # 广播到 128 个 head
        key = torch.cat([k, rope.to(k.dtype)], dim=-1)
        return key, v


def get_init_inputs():
    return [K_DIM, V_DIM, EMB]


def get_inputs():
    kv = torch.randn(SEQ, BATCH, NHEADS, K_DIM + V_DIM).to(torch.bfloat16)
    k_pos_emb = torch.randn(SEQ, BATCH, 1, EMB).to(torch.bfloat16)
    theta = 1.0 / (10000.0 ** (torch.arange(0, EMB, 2, dtype=torch.float32) / EMB))
    pos = torch.arange(SEQ, dtype=torch.float32)
    ang = pos[:, None] * theta[None, :]              # [S, emb/2]
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1).view(SEQ, 1, 1, EMB)
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1).view(SEQ, 1, 1, EMB)
    return [kv, k_pos_emb, cos, sin]
