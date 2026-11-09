"""Reference case for rotary_fwd_q_kernel.

Source: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent op: ApplyMLARotaryEmbQ
(megatron/core/fusions/fused_mla_yarn_rope_apply.py).

The kernel applies YARN RoPE in place to the last emb_dim of MLA query.
The leading qk_head_dim is left unchanged.

  rope = q[..., qk_head_dim:]          # [seq, batch, heads, 64]
  x1, x2 = rope[..., 0::2], rope[..., 1::2]
  left  = x1 * cos[..., :32] - x2 * sin[..., :32]
  right = x2 * cos[..., 32:] + x1 * sin[..., 32:]
  q[..., qk_head_dim:] = cat(left, right)

cos/sin are [seq, 1, 1, emb_dim] and are already the local CP slice.
For this sbhd launch the kernel indexes them with the local sequence
position (pid_m // batch). The CP zigzag split happens when the cos/sin
cache is built, not inside rotary_fwd_q_kernel.

Shape from the 1188B MLA config and the recorded launch:
  grid (16384, 2, 1), block (128, 1, 1), shared memory 2816 bytes.
  grid.x = seq * batch = 16384 (seq 131072 / CP 8, batch 1).
  grid.y = cdiv(heads, BLOCK_H) = 2, heads = 128, so BLOCK_H = 64.
  qk_head_dim = 128, qk_pos_emb_head_dim = 64, so Q is [16384, 1, 128, 192].
  128 launches, duration 429..455 us. Full recompute runs this forward twice.
  TP is 1, so all 128 heads sit on this rank.
"""

from __future__ import annotations

import torch


SEQ = 16384
BATCH = 1
HEADS = 128
QK_HEAD_DIM = 128
EMB_DIM = 64


def reference_rotary_fwd_q(
    q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """YARN RoPE on the tail of Q. Returns a new tensor; the kernel is in place.

    q:   [seq, batch, heads, qk_head_dim + emb_dim], bf16
    cos, sin: [seq, 1, 1, emb_dim] or [seq, emb_dim]
    """
    rope = q[..., QK_HEAD_DIM:].float()
    x1 = rope[..., 0::2]
    x2 = rope[..., 1::2]
    cos_f = cos.float().reshape(cos.shape[0], EMB_DIM)
    sin_f = sin.float().reshape(sin.shape[0], EMB_DIM)
    half = EMB_DIM // 2
    # [seq, 1, 1, half] broadcasts over batch and heads.
    cos_left = cos_f[:, None, None, :half]
    cos_right = cos_f[:, None, None, half:]
    sin_left = sin_f[:, None, None, :half]
    sin_right = sin_f[:, None, None, half:]
    left = x1 * cos_left - x2 * sin_left
    right = x2 * cos_right + x1 * sin_right
    out = q.clone()
    out[..., QK_HEAD_DIM:] = torch.cat((left, right), dim=-1).to(q.dtype)
    return out


def make_inputs(seed: int = 0, device: torch.device | str = "cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    q = torch.randn(
        SEQ, BATCH, HEADS, QK_HEAD_DIM + EMB_DIM, generator=g, dtype=torch.bfloat16
    )
    # Yarn builds emb as cat(freqs, freqs), so the two halves match.
    freq = torch.randn(SEQ, EMB_DIM // 2, generator=g, dtype=torch.float32)
    emb = torch.cat((freq, freq), dim=-1)
    cos = emb.cos().to(torch.bfloat16).view(SEQ, 1, 1, EMB_DIM)
    sin = emb.sin().to(torch.bfloat16).view(SEQ, 1, 1, EMB_DIM)
    return q.to(device), cos.to(device), sin.to(device)


def _manual_one(q, cos, sin, seq_idx: int, head: int) -> torch.Tensor:
    rope = q[seq_idx, 0, head, QK_HEAD_DIM:].float()
    x1 = rope[0::2]
    x2 = rope[1::2]
    c = cos[seq_idx].float().reshape(EMB_DIM)
    s = sin[seq_idx].float().reshape(EMB_DIM)
    half = EMB_DIM // 2
    return torch.cat((x1 * c[:half] - x2 * s[:half], x2 * c[half:] + x1 * s[half:]))


def run_case(device: str = "cpu", seed: int = 0) -> None:
    q, cos, sin = make_inputs(seed=seed, device=device)
    got = reference_rotary_fwd_q(q, cos, sin)
    # Content half is not part of the rotary update.
    content_delta = (got[..., :QK_HEAD_DIM] - q[..., :QK_HEAD_DIM]).abs().max().item()
    max_abs = 0.0
    for seq_idx, head in ((0, 0), (SEQ // 2, HEADS // 2), (SEQ - 1, HEADS - 1)):
        manual = _manual_one(q, cos, sin, seq_idx, head).to(q.dtype)
        got_rope = got[seq_idx, 0, head, QK_HEAD_DIM:]
        max_abs = max(max_abs, (got_rope - manual).abs().max().item())
    print(f"device={device} q={tuple(q.shape)} {q.dtype}")
    print(f"content_delta={content_delta:.6e} rope_max_abs={max_abs:.6e}")
    if content_delta != 0 or max_abs > 1e-3:
        raise SystemExit("RoPE reference disagrees with the explicit formula")
    print("reference check passed")


if __name__ == "__main__":
    run_case("cpu")
    if hasattr(torch, "musa") and torch.musa.is_available():
        run_case("musa")
