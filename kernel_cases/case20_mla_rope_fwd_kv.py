"""case20（真实形状版）— MLA YaRN RoPE forward for K/V: rotary_fwd_kv_kernel

形状来源（唯一权威）: 021-1188B-real-ops-11-11-record.xlsx
  trace: x1w_1188B_128k_ac-full_rank-8.json（seq 131072 / CP 8 / 每卡 16384 tokens）
  记录: kv(16384,1,128,256) + k_pos_emb(16384,1,1,64) + cos,sin(16384,1,1,64)
        -> key(16384,1,128,192) + value(16384,1,128,128)
  profiler 真实平均: 2.09 ms
  自洽校核: 读 1.07GB + 写 1.34GB ≈ 2.42GB/次。
注意: 旧 kernel_cases 版 SEQ=4096，作废；语义与 rope 旋转结构不变，仅 S 4096->16384。
"""

import torch
import torch.nn as nn

# ---- real per-rank dims (xlsx record, 128k seq / CP 8) ----
SEQ = 16384
BATCH = 1
NHEADS = 128
K_DIM = 128
V_DIM = 128
EMB = 64

CASE = {
    "id": 20,
    "name": "mla_rope_fwd_kv",
    "kernel": "rotary_fwd_kv_kernel",
    "csv_row": 20,
    "profiler_avg_ms": 2.09,
    "shape_provenance": "021-1188B-real-ops-11-11-record.xlsx (x1w_1188B_128k_ac-full_rank-8.json)",
    "backward": False,
}


class Model(nn.Module):
    """key = cat([kv[..., :K_DIM], rope(k_pos_emb)], -1), value = kv[..., K_DIM:]（half-split 旋转，按 head 广播）。"""

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
        x = k_pos_emb.float()
        x1, x2 = x[..., 0::2], x[..., 1::2]
        c1, s1 = cos[..., :half], sin[..., :half]
        c2, s2 = cos[..., half:], sin[..., half:]
        left = x1 * c1 - x2 * s1
        right = x2 * c2 + x1 * s2
        rope = torch.cat([left, right], dim=-1)
        rope = rope.expand(-1, -1, k.shape[-2], -1)
        key = torch.cat([k, rope.to(k.dtype)], dim=-1)
        return key, v


def get_init_inputs():
    return [K_DIM, V_DIM, EMB]


def get_inputs():
    torch.manual_seed(20261120)
    kv = torch.randn(SEQ, BATCH, NHEADS, K_DIM + V_DIM).to(torch.bfloat16)
    k_pos_emb = torch.randn(SEQ, BATCH, 1, EMB).to(torch.bfloat16)
    theta = 1.0 / (10000.0 ** (torch.arange(0, EMB, 2, dtype=torch.float32) / EMB))
    pos = torch.arange(SEQ, dtype=torch.float32)
    ang = pos[:, None] * theta[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1).view(SEQ, 1, 1, EMB)
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1).view(SEQ, 1, 1, EMB)
    return [kv, k_pos_emb, cos, sin]
