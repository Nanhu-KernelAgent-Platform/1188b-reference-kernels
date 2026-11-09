"""Reference case for rotary_bwd_kv_kernel.

Source: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent op: ApplyMLARotaryEmbKV Backward
(megatron/core/fusions/fused_mla_yarn_rope_apply.py).

Forward (rotary_fwd_kv_kernel) splits MLA's packed KV, copies the content
halves, and applies YARN RoPE to the shared k_pos_emb once, then writes the
same rotated emb onto every head of o_key:

  o_key[..., :k_dim]              = kv[..., :k_dim]
  o_value                         = kv[..., k_dim:]
  x1, x2 = emb[..., 0::2], emb[..., 1::2]
  left  = x1 * cos[..., :32] - x2 * sin[..., :32]
  right = x2 * cos[..., 32:] + x1 * sin[..., 32:]
  o_key[..., k_dim:]              = broadcast_heads(cat(left, right))

Backward inverts that. Content gradients copy straight through. The emb
gradient sums the rope half of dK over heads, then applies the adjoint of
the YARN rotate (stored layout is cat(left, right), not reinterleaved):

  d_kv[..., :k_dim] = dk[..., :k_dim]
  d_kv[..., k_dim:] = dv
  L = sum_heads(dk[..., k_dim : k_dim + half])
  R = sum_heads(dk[..., k_dim + half :])
  dx1 = L * cos_left  + R * sin_right
  dx2 = -L * sin_left + R * cos_right
  d_emb = interleave(dx1, dx2)

cos/sin are the local CP slice. For this sbhd launch token_idx = pid_m //
batch. Only program pid_head == 0 writes d_emb; the traced launch has
grid.y = 1 so BLOCK_H = 128 covers all heads in one block.

Shape from the 1188B MLA config and the recorded launch:
  k_dim = qk_head_dim = 128, v_dim = v_head_dim = 128, emb_dim = 64.
  heads = 128, seq = 16384 (131072 / CP 8), batch = 1.
  dk:    [16384, 1, 128, 192] bf16   # emb_dim + k_dim
  dv:    [16384, 1, 128, 128] bf16
  d_kv:  [16384, 1, 128, 256] bf16   # k_dim + v_dim
  d_emb: [16384, 1, 1, 64] bf16

Launch recorded on MTT S5000:
  grid (16384, 1, 1), block (128, 1, 1), shared memory 768 bytes.
  grid.x = seq * batch. grid.y = cdiv(heads, BLOCK_H) = 1 => BLOCK_H = 128.
  64 launches (one KV RoPE backward per layer).
  Duration 1846.690..1917.571 us (median ~1881 us).
"""

from __future__ import annotations

import torch


SEQ = 16384
BATCH = 1
HEADS = 128
K_DIM = 128
V_DIM = 128
EMB_DIM = 64
HALF = EMB_DIM // 2


def reference_rotary_bwd_kv(
    dk: torch.Tensor,
    dv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of ApplyMLARotaryEmbKV. Returns (d_kv, d_emb).

    dk:  [seq, batch, heads, emb_dim + k_dim], bf16
    dv:  [seq, batch, heads, v_dim], bf16
    cos/sin: [seq, 1, 1, emb_dim] or [seq, emb_dim]
    """
    if dk.shape[-1] != EMB_DIM + K_DIM or dv.shape[-1] != V_DIM:
        raise ValueError(f"shapes dk {tuple(dk.shape)} dv {tuple(dv.shape)}")
    d_kv = dk.new_empty(*dk.shape[:-1], K_DIM + V_DIM)
    d_kv[..., :K_DIM] = dk[..., :K_DIM]
    d_kv[..., K_DIM:] = dv

    left = dk[..., K_DIM : K_DIM + HALF].float().sum(dim=2)  # [seq, batch, half]
    right = dk[..., K_DIM + HALF :].float().sum(dim=2)
    cos_f = cos.float().reshape(cos.shape[0], EMB_DIM)
    sin_f = sin.float().reshape(sin.shape[0], EMB_DIM)
    # [seq, 1, half] broadcasts over batch.
    cos_left = cos_f[:, None, :HALF]
    cos_right = cos_f[:, None, HALF:]
    sin_left = sin_f[:, None, :HALF]
    sin_right = sin_f[:, None, HALF:]
    dx1 = left * cos_left + right * sin_right
    dx2 = -left * sin_left + right * cos_right
    # join(x1, x2) -> interleaved emb gradient.
    d_emb = torch.stack((dx1, dx2), dim=-1).reshape(*dx1.shape[:-1], EMB_DIM)
    d_emb = d_emb.to(dk.dtype).unsqueeze(2)  # [seq, batch, 1, emb_dim]
    return d_kv, d_emb


def make_inputs(seed: int = 0, device: torch.device | str = "cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    dk = torch.randn(SEQ, BATCH, HEADS, EMB_DIM + K_DIM, generator=g, dtype=torch.bfloat16)
    dv = torch.randn(SEQ, BATCH, HEADS, V_DIM, generator=g, dtype=torch.bfloat16)
    freq = torch.randn(SEQ, HALF, generator=g, dtype=torch.float32)
    emb = torch.cat((freq, freq), dim=-1)
    cos = emb.cos().to(torch.bfloat16).view(SEQ, 1, 1, EMB_DIM)
    sin = emb.sin().to(torch.bfloat16).view(SEQ, 1, 1, EMB_DIM)
    return dk.to(device), dv.to(device), cos.to(device), sin.to(device)


def _manual_one(dk, dv, cos, sin, seq_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit one-token check: d_kv for head 0 and the shared d_emb."""
    d_kv_h0 = torch.cat((dk[seq_idx, 0, 0, :K_DIM], dv[seq_idx, 0, 0]), dim=-1)
    left = dk[seq_idx, 0, :, K_DIM : K_DIM + HALF].float().sum(dim=0)
    right = dk[seq_idx, 0, :, K_DIM + HALF :].float().sum(dim=0)
    c = cos[seq_idx].float().reshape(EMB_DIM)
    s = sin[seq_idx].float().reshape(EMB_DIM)
    dx1 = left * c[:HALF] + right * s[HALF:]
    dx2 = -left * s[:HALF] + right * c[HALF:]
    d_emb = torch.stack((dx1, dx2), dim=-1).reshape(EMB_DIM)
    return d_kv_h0, d_emb.to(dk.dtype)


def _check_adjoint_identity() -> None:
    """For a single head, <dK_rope, fwd(emb)> ~= <d_emb, emb> with random cos/sin."""
    g = torch.Generator(device="cpu")
    g.manual_seed(1)
    emb = torch.randn(EMB_DIM, generator=g, dtype=torch.float32)
    freq = torch.randn(HALF, generator=g, dtype=torch.float32)
    ang = torch.cat((freq, freq))
    c = ang.cos()
    s = ang.sin()
    x1, x2 = emb[0::2], emb[1::2]
    left = x1 * c[:HALF] - x2 * s[:HALF]
    right = x2 * c[HALF:] + x1 * s[HALF:]
    y = torch.cat((left, right))

    d_left = torch.randn(HALF, generator=g, dtype=torch.float32)
    d_right = torch.randn(HALF, generator=g, dtype=torch.float32)
    dx1 = d_left * c[:HALF] + d_right * s[HALF:]
    dx2 = -d_left * s[:HALF] + d_right * c[HALF:]
    d_emb = torch.stack((dx1, dx2), dim=-1).reshape(EMB_DIM)
    lhs = (torch.cat((d_left, d_right)) * y).sum()
    rhs = (d_emb * emb).sum()
    if (lhs - rhs).abs().item() > 1e-5:
        raise SystemExit(f"RoPE adjoint identity failed: {lhs.item()} vs {rhs.item()}")


def run_case(device: str = "cpu", seed: int = 0) -> None:
    dk, dv, cos, sin = make_inputs(seed=seed, device=device)
    d_kv, d_emb = reference_rotary_bwd_kv(dk, dv, cos, sin)

    content_k = (d_kv[..., :K_DIM] - dk[..., :K_DIM]).abs().max().item()
    content_v = (d_kv[..., K_DIM:] - dv).abs().max().item()
    max_abs = 0.0
    for seq_idx in (0, SEQ // 2, SEQ - 1):
        manual_kv, manual_emb = _manual_one(dk, dv, cos, sin, seq_idx)
        err_kv = (d_kv[seq_idx, 0, 0] - manual_kv).abs().max().item()
        err_emb = (d_emb[seq_idx, 0, 0] - manual_emb).abs().max().item()
        max_abs = max(max_abs, err_kv, err_emb)
    print(
        f"device={device} dk={tuple(dk.shape)} {dk.dtype} "
        f"d_kv={tuple(d_kv.shape)} {d_kv.dtype} "
        f"d_emb={tuple(d_emb.shape)} {d_emb.dtype}"
    )
    print(
        f"content_k_delta={content_k:.6e} content_v_delta={content_v:.6e} "
        f"rope_max_abs={max_abs:.6e}"
    )
    if content_k != 0 or content_v != 0 or max_abs > 1e-3:
        raise SystemExit("KV RoPE backward disagrees with the explicit formula")
    print("reference check passed")


if __name__ == "__main__":
    _check_adjoint_identity()
    run_case("cpu")
    if hasattr(torch, "musa") and torch.musa.is_available():
        run_case("musa")
