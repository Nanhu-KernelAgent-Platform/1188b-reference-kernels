"""case01 — MLA/FlashAttention backward（causal, softmax fp32, bf16）:
          musa_asm_bf16tce_flash_atten_bwd_512_64x128x_qk_192_v_128_ws_causal_persistence

CSV   : 021-1188B-real-ops.csv / sheet 测试对象2 / 第 1 行
kernel: musa_asm_bf16tce_flash_atten_bwd_512_64x128x_qk_192_v_128_ws_causal_persistence
上层   : AttnFuncWithCPAndQKVOA2ABackward（transformer_engine/pytorch/attention.py:4134 的 backward；
         MUSA 启动点 transformer_engine/musa/pytorch/csrc/extensions/attention.mu:573,617）

真实形状（per-rank，TP=1 / CP=1 / micro-batch=1 / sbhd）:
  q     [4096, 1, 128, 192]  bf16
  k     [4096, 1, 128, 192]  bf16
  v     [4096, 1, 128, 128]  bf16
  o     [4096, 1, 128, 128]  bf16      # forward 输出
  do    [4096, 1, 128, 128]  bf16      # 上游对 o 的梯度
  ----
  dq    [4096, 1, 128, 192]  bf16
  dk    [4096, 1, 128, 192]  bf16
  dv    [4096, 1, 128, 128]  bf16

语义（标准 flash-attention 反向）:
  p    = softmax(q k^T * scale + causal)                 (fp32)
  dv  += p^T do
  dp   = do v^T
  ds   = p * (dp - rowsum(dp * p))                       (softmax 雅可比)
  dq   = ds k * scale
  dk  += ds^T q * scale
scale = 1/sqrt(192)。两套参考实现（默认 sdpa）:
  * impl="sdpa"     : torch SDPA + autograd（官方 flash-attn 反向路径，基准性能用这个）
  * impl="explicit" : 分块解析反向（自己重算 p，不物化 S x S），用于交叉验证
forward 输出 o 只是真实 ABI 的一部分；两种实现都不读取它（FA2 形式只需 q,k,v,do）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- real per-rank dims (pretrain_1188B_formal_training.sh) ----
SEQ = 4096
BATCH = 1
NHEADS = 128
DQK = 192
DV = 128

CASE = {
    "id": 1,
    "name": "mla_flash_attn_bwd_causal",
    "kernel": "musa_asm_bf16tce_flash_atten_bwd_512_64x128x_qk_192_v_128_ws_causal_persistence",
    "upstream": "AttnFuncWithCPAndQKVOA2ABackward",
    "csv_row": 1,
    "source": "transformer_engine/pytorch/attention.py:4134 (backward); "
              "transformer_engine/musa/pytorch/csrc/extensions/attention.mu:573,617",
    "dtype": "bf16 in/out, fp32 softmax",
    "provenance": "形状由源码+配置直接给出（高）；kernel 名里的 qk_192/v_128 与 MLA 配置一致（高）",
    "backward": False,
}


def flash_attn_bwd(q, k, v, o, do, scale, causal=True, chunk=256):
    """返回 (dq, dk, dv)。按 q 分块重算 p，显存 O(chunk * S)。"""
    s, b, h, dqk = q.shape
    dv_dim = v.shape[-1]
    qp = q.permute(1, 2, 0, 3).contiguous()
    kp = k.permute(1, 2, 0, 3).contiguous()
    vp = v.permute(1, 2, 0, 3).contiguous()
    dop = do.permute(1, 2, 0, 3).contiguous()
    kt = kp.transpose(-1, -2)
    vt = vp.transpose(-1, -2)

    dq = torch.zeros_like(qp)
    dk = torch.zeros((b, h, s, dqk), dtype=torch.float32, device=q.device)
    dvv = torch.zeros((b, h, s, dv_dim), dtype=torch.float32, device=q.device)
    idx = torch.arange(s, device=q.device)

    for i0 in range(0, s, chunk):
        i1 = min(i0 + chunk, s)
        qc = qp[:, :, i0:i1]
        doc = dop[:, :, i0:i1]
        sc = torch.matmul(qc, kt).float() * scale
        if causal:
            keep = idx[None, :] <= idx[i0:i1][:, None]
            sc = sc.masked_fill(~keep, float("-inf"))
        p = torch.softmax(sc, dim=-1)                       # fp32, [-1] 行除 mask 外和为 1
        pb = p.to(q.dtype)

        dvv += torch.matmul(pb.transpose(-1, -2), doc)       # [B,H,S,Dv]
        dp = torch.matmul(doc, vt).float()                   # [B,H,m,S]
        row = (dp * p).sum(dim=-1, keepdim=True)
        ds = p * (dp - row)                                  # fp32
        dsb = ds.to(q.dtype)
        dq[:, :, i0:i1] = torch.matmul(dsb, kp) * scale
        dk += torch.matmul(dsb.transpose(-1, -2), qc) * scale
        del sc, p, pb, dp, ds, dsb

    return (dq.permute(2, 0, 1, 3).contiguous().to(q.dtype),
            dk.permute(2, 0, 1, 3).contiguous().to(q.dtype),
            dvv.permute(2, 0, 1, 3).contiguous().to(v.dtype))


class Model(nn.Module):
    """impl="sdpa"：对 torch SDPA 做 autograd（官方 flash-attn 反向）；"explicit"：分块解析反向。"""

    def __init__(self, chunk=256, impl="sdpa"):
        super().__init__()
        self.chunk = chunk
        self.impl = impl

    def forward(self, q, k, v, o, do):
        scale = q.shape[-1] ** -0.5
        if self.impl == "sdpa":
            # 外层可能处于 torch.no_grad()，这里显式打开 grad 以便对 SDPA 求导
            with torch.enable_grad():
                leaves = [x.permute(1, 2, 0, 3).contiguous().detach().requires_grad_(True)
                          for x in (q, k, v)]
                out = F.scaled_dot_product_attention(*leaves, is_causal=True, scale=scale)
                g = torch.autograd.grad(out, leaves, do.permute(1, 2, 0, 3).contiguous())
            return tuple(gi.permute(2, 0, 1, 3).contiguous().to(q.dtype) for gi in g)
        return flash_attn_bwd(q, k, v, o, do, scale, causal=True, chunk=self.chunk)


def get_init_inputs():
    return [256, "sdpa"]


def get_inputs():
    q = torch.randn(SEQ, BATCH, NHEADS, DQK).to(torch.bfloat16)
    k = torch.randn(SEQ, BATCH, NHEADS, DQK).to(torch.bfloat16)
    v = torch.randn(SEQ, BATCH, NHEADS, DV).to(torch.bfloat16)
    o = torch.randn(SEQ, BATCH, NHEADS, DV).to(torch.bfloat16)
    do = torch.randn(SEQ, BATCH, NHEADS, DV).to(torch.bfloat16) * 0.01
    return [q, k, v, o, do]
