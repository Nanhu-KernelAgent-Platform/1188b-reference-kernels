"""Reference case for

musa::dnn::preprocess_delta_kernel<__mt_bfloat16, false, 1024, 128, 64, 128>

Source: x1w_1188B_128k_ac-full_rank-8.json, rank 8, one training step.
Parent op: AttnFuncWithCPAndQKVOA2ABackward. This is the FlashAttention
backward preprocess: the per-token dot product of out and dout.

  delta[h, s] = sum_d out[s, h, d] * dout[s, h, d]

  out and dout are bf16. The sum is fp32. The causal mask is not used;
  every head element participates. The template bool is the fixed-length
  specialization (the kernel still receives a cu_seqlens pointer).

Shape after the Ulysses QKVO all-to-all. CP=8, 128 heads, v_head_dim=128:
  local sequence before A2A is 131072 / 8 = 16384.
  after A2A this rank holds the full sequence and 128 / 8 = 16 heads.
  out, dout: [131072, 16, 128]
  delta:     [16, 131072] fp32, the same [head, seq] layout as softmax LSE.

Launch recorded on MTT S5000:
  grid (1024, 16, 1), block (1024, 1, 1), shared memory 4224 bytes.
  grid.y = 16 heads.
  grid.x = 131072 / 128 = 1024, so each block covers 128 tokens of one head.
  Template integers are the compiled tile: 1024 threads, 128-token tile,
  head dim 128. The kernel is launched 64 times (one attention backward
  per layer call; the forward is run twice by full recompute).
  Duration 896..948 us.
"""

from __future__ import annotations

import torch


SEQ = 131072
HEADS = 16
DIM = 128


def reference_preprocess_delta(out: torch.Tensor, dout: torch.Tensor) -> torch.Tensor:
    """delta[h, s] = sum_d out[s, h, d] * dout[s, h, d], accumulated in fp32."""
    return (out.float() * dout.float()).sum(dim=-1).transpose(0, 1).contiguous()


def make_inputs(seed: int = 0, device: torch.device | str = "cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    out = torch.randn(SEQ, HEADS, DIM, generator=g, dtype=torch.bfloat16)
    dout = torch.randn(SEQ, HEADS, DIM, generator=g, dtype=torch.bfloat16)
    return out.to(device), dout.to(device)


def _check_ones() -> None:
    out = torch.ones(4, 2, DIM, dtype=torch.bfloat16)
    dout = torch.ones_like(out)
    delta = reference_preprocess_delta(out, dout)
    if delta.shape != (2, 4) or not torch.equal(delta, torch.full_like(delta, float(DIM))):
        raise SystemExit("delta of ones is not the head dim")


def run_case(device: str = "cpu", seed: int = 0) -> None:
    out, dout = make_inputs(seed=seed, device=device)
    got = reference_preprocess_delta(out, dout)
    max_abs = 0.0
    for seq_idx, head in ((0, 0), (SEQ // 2, HEADS // 2), (SEQ - 1, HEADS - 1)):
        manual = (out[seq_idx, head].float() * dout[seq_idx, head].float()).sum()
        max_abs = max(max_abs, (got[head, seq_idx] - manual).abs().item())
    print(f"device={device} out={tuple(out.shape)} {out.dtype} delta={tuple(got.shape)} {got.dtype}")
    print(f"max_abs={max_abs:.6e}")
    if got.shape != (HEADS, SEQ) or max_abs > 1e-4:
        raise SystemExit("delta disagrees with the explicit dot product")
    print("reference check passed")


if __name__ == "__main__":
    _check_ones()
    run_case("cpu")
    if hasattr(torch, "musa") and torch.musa.is_available():
        run_case("musa")
